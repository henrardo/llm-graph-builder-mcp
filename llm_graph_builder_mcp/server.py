#!/usr/bin/env python3
"""
MCP Server for Neo4j LLM Graph Builder

Provides tools to build knowledge graphs from unstructured text and documents using LLMs.
"""

import json
import logging
import os
from typing import Any, Optional

import httpx
from fastmcp import FastMCP
from pydantic import Field

# Configure logging with file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/llm_graph_builder_mcp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
GRAPH_BUILDER_URL = os.getenv("GRAPH_BUILDER_URL", "http://localhost:8000")
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Initialize FastMCP server
mcp = FastMCP("llm-graph-builder", dependencies=["httpx", "pydantic"])

# Create a shared HTTP client with extended timeout
http_client = httpx.AsyncClient(timeout=600.0)  # 10 min timeout for large documents


def create_form_data(**kwargs) -> dict[str, Any]:
    """Helper to create form data, filtering out None values.
    Empty strings are preserved (backend needs them for optional schema params)."""
    return {k: str(v) for k, v in kwargs.items() if v is not None}


def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    url_lower = url.lower()
    # Check file extension
    if url_lower.endswith('.pdf'):
        return True
    # Check if 'pdf' is in URL path or query (common for CDN/cloud storage)
    if '.pdf?' in url_lower or '/pdf/' in url_lower or 'pdf' in url_lower.split('/')[-1]:
        return True
    return False


async def process_pdf_url(
    url: str,
    model: str,
    allowed_nodes: Optional[str],
    allowed_relationships: Optional[str],
    enable_communities: bool,
    extract_bibliographic_info: bool = False
) -> str:
    """Download PDF from URL and upload it to backend for processing."""
    logger.info(f"Processing PDF URL: {url}")
    
    try:
        # Step 1: Download PDF
        logger.info("Downloading PDF...")
        download_response = await http_client.get(url, follow_redirects=True)
        download_response.raise_for_status()
        pdf_content = download_response.content
        pdf_size = len(pdf_content)
        logger.info(f"Downloaded PDF: {pdf_size} bytes")
        
        # Extract filename from URL
        import re
        from urllib.parse import unquote, urlparse
        
        parsed = urlparse(url)
        path = unquote(parsed.path)
        
        # Try to get filename from URL
        if 'filename=' in url:
            # Extract from query param (e.g., CloudFront)
            match = re.search(r'filename=([^&]+)', url)
            if match:
                file_name = unquote(match.group(1))
            else:
                file_name = path.split('/')[-1] or 'document.pdf'
        else:
            file_name = path.split('/')[-1] or 'document.pdf'
        
        # Ensure .pdf extension
        if not file_name.lower().endswith('.pdf'):
            file_name += '.pdf'
        
        logger.info(f"Extracted filename: {file_name}")
        
        # Step 2: Upload to backend via /upload endpoint
        logger.info("Uploading PDF to backend...")
        
        files = {
            'file': (file_name, pdf_content, 'application/pdf')
        }
        
        upload_data = {
            'chunkNumber': '1',
            'totalChunks': '1',
            'originalname': file_name,
            'model': model,
            'uri': NEO4J_URI,
            'userName': NEO4J_USERNAME,
            'password': NEO4J_PASSWORD,
            'database': NEO4J_DATABASE
        }
        
        upload_response = await http_client.post(
            f"{GRAPH_BUILDER_URL}/upload",
            files=files,
            data=upload_data
        )
        upload_response.raise_for_status()
        upload_result = upload_response.json()
        
        logger.info(f"Upload response: {upload_result}")
        
        if upload_result.get("status") != "Success":
            raise Exception(f"Upload failed: {upload_result.get('message')}")
        
        # The file is now uploaded and source node created
        # Now extract entities
        logger.info("Extracting entities from uploaded PDF...")
        
        # Convert None to single space for schema params
        nodes_param = allowed_nodes if allowed_nodes else " "
        rels_param = allowed_relationships if allowed_relationships else " "
        
        # Build additional instructions for bibliographic extraction
        additional_instructions = None
        if extract_bibliographic_info:
            additional_instructions = """
ACADEMIC PAPER EXTRACTION INSTRUCTIONS:

Extract comprehensive bibliographic information and citations:

1. DOCUMENT METADATA:
   - Extract: Author(s), Title, Journal/Venue, Publication Year, DOI, Abstract
   - Create nodes: Person (authors), Publication (this paper), Journal, Year
   - Relationships: Person-AUTHORED->Publication, Publication-PUBLISHED_IN->Journal, Publication-PUBLISHED_ON->Year

2. CITED WORKS (References section):
   - For each reference: Extract author(s), title, journal, year
   - Create nodes: Person (cited authors), Publication (cited works), Journal, Year
   - Relationships: Publication-CITES->Publication, Person-AUTHORED->Publication

3. KEY CONCEPTS & TOPICS:
   - Extract: Research topics, methods, theories, geographical locations, time periods
   - Create nodes: Concept, Method, Theory, Location, Date/Period
   - Relationships: Publication-DISCUSSES->Concept, Publication-USES_METHOD->Method, Publication-FOCUSES_ON->Location

4. INTRA-PAPER STRUCTURE:
   - Extract: Sections (Introduction, Methods, Results, Discussion)
   - Relationships: Publication-HAS_SECTION->Section, Section-DISCUSSES->Concept

CRITICAL INSTRUCTIONS:
- Pay special attention to the References/Bibliography section
- Extract author names exactly as written
- Preserve journal/venue names completely
- Extract publication years as numeric values
- Create CITES relationships between papers
- Link authors to their papers with AUTHORED relationships
- Link papers to journals with PUBLISHED_IN relationships

Focus on creating a rich citation network showing academic lineage and knowledge flow.
"""
        
        extract_data = create_form_data(
            uri=NEO4J_URI,
            userName=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            model=model,
            file_name=file_name,
            source_type='local file',  # Backend treats uploaded files as local
            allowedNodes=nodes_param,
            allowedRelationship=rels_param,
            token_chunk_size=500,
            chunk_overlap=50,
            chunks_to_combine=1,
            additional_instructions=additional_instructions if additional_instructions else None
        )
        
        extract_response = await http_client.post(
            f"{GRAPH_BUILDER_URL}/extract",
            data=extract_data
        )
        extract_response.raise_for_status()
        extract_result = extract_response.json()
        
        if extract_result.get("status") == "Failed":
            error_msg = extract_result.get("message", "Unknown error")
            raise Exception(f"Entity extraction failed: {error_msg}")
        
        # Build response
        data = extract_result.get("data", {})
        response_data = {
            "status": "Success",
            "message": f"Successfully built knowledge graph from PDF: {file_name}",
            "source_url": url,
            "file_name": file_name,
            "file_size": pdf_size,
            "nodes_created": data.get("nodeCount", 0),
            "relationships_created": data.get("relationshipCount", 0),
            "processing_time": data.get("processingTime", 0)
        }
        
        # Handle community detection if enabled
        if enable_communities:
            logger.info("Running community detection...")
            post_processing_data = create_form_data(
                uri=NEO4J_URI,
                userName=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE,
                tasks=json.dumps(["enable_communities"])
            )
            
            post_processing_response = await http_client.post(
                f"{GRAPH_BUILDER_URL}/post_processing",
                data=post_processing_data
            )
            post_processing_response.raise_for_status()
            post_processing_result = post_processing_response.json()
            
            if post_processing_result.get("status") == "Success":
                logger.info("Community detection: Success")
                community_data = post_processing_result.get("data", [])
                if community_data:
                    for item in community_data:
                        if item.get("filename") == file_name:
                            response_data["communities_created"] = item.get("communityNodeCount", 0)
                            response_data["community_relationships"] = item.get("communityRelCount", 0)
                            break
            else:
                logger.warning(f"Community detection failed: {post_processing_result.get('message')}")
                response_data["community_detection_status"] = "Failed"
        
        return json.dumps(response_data, indent=2)
        
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error while processing PDF: {e.response.status_code} - {e.response.text[:500]}"
        logger.error(error_msg)
        return json.dumps({"status": "Failed", "message": error_msg})
    except Exception as e:
        error_msg = f"Failed to process PDF URL: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"status": "Failed", "message": error_msg})


@mcp.tool()
async def build_knowledge_graph_from_url(
    url: str = Field(
        ..., 
        description="URL to extract knowledge graph from (web page, Wikipedia, YouTube, or PDF)"
    ),
    model: str = Field(
        default="openai_gpt_4.1", 
        description="LLM model to use for extraction (default: openai_gpt_4.1)"
    ),
    allowed_nodes: Optional[str] = Field(
        None, 
        description=(
            "Optional: Comma-separated list of allowed entity types to extract. "
            "If not specified, all entity types will be extracted. "
            "Common types: Person, Organization, Location, Event, Date, Work, Concept, Technology. "
            "Example: 'Person,Organization,Location,Event'"
        )
    ),
    allowed_relationships: Optional[str] = Field(
        None, 
        description=(
            "Optional: Define allowed relationships as comma-separated TRIPLES (source,relation,target). "
            "Each triple defines one allowed relationship type between two entity types. "
            "Format: 'SourceType,RELATIONSHIP,TargetType,SourceType,RELATIONSHIP,TargetType,...' "
            "\n\nExamples:"
            "\n- Historical: 'Person,BORN_IN,Location,Person,DIED_IN,Location,Person,CREATED,Work,Work,PUBLISHED_IN,Location'"
            "\n- Business: 'Person,WORKS_FOR,Organization,Person,FOUNDED,Organization,Product,MADE_BY,Organization'"
            "\n- Academic: 'Person,AUTHORED,Publication,Person,AFFILIATED_WITH,Organization,Publication,CITES,Publication'"
            "\n\nIMPORTANT: Source and target types MUST be in the allowed_nodes list. "
            "If not specified, ALL relationships between allowed nodes will be extracted."
        )
    ),
    enable_communities: bool = Field(
        default=False,
        description=(
            "Optional: Enable community detection after extraction (default: False). "
            "When True, runs hierarchical community detection using the Leiden algorithm to identify "
            "clusters of densely connected entities. Creates intermediate Community nodes and "
            "IN_COMMUNITY relationships. Useful for discovering groups, topics, or organizational structures "
            "in the knowledge graph."
        )
    ),
    extract_bibliographic_info: bool = Field(
        default=False,
        description=(
            "Optional: Extract bibliographic information from academic papers (default: False). "
            "When True, instructs the LLM to specifically extract: "
            "\n- Document metadata (authors, title, journal, year, DOI)"
            "\n- Citations and references (cited works, their authors, journals)"
            "\n- Relationships (AUTHORED, CITES, PUBLISHED_IN, DISCUSSES)"
            "\n- Academic concepts, methods, and theories"
            "\nPerfect for research papers, academic articles, and building citation networks. "
            "Highly recommended for Zotero integrations and scholarly work."
        )
    ),
) -> str:
    """
    Build a knowledge graph from a URL source (web page, Wikipedia, YouTube video).
    
    This tool performs two steps:
    1. Scans the URL and creates a source node in Neo4j
    2. Extracts entities and relationships from the content
    
    Args:
        url: URL of the source (e.g., https://en.wikipedia.org/wiki/Hartmann_Schedel)
        model: LLM model to use (openai_gpt_4.1_2025_04_14, openai_gpt_4o, openai_gpt_4o_mini, etc.)
        allowed_nodes: Optional comma-separated entity types to extract
        allowed_relationships: Optional comma-separated relationship types to extract
    
    Returns:
        JSON string with extraction results including node and relationship counts
    """
    try:
        # DEBUG: Log what we received
        logger.info(f"=== MCP RECEIVED ===")
        logger.info(f"URL parameter: {url}")
        logger.info(f"Model: {model}")
        logger.info(f"===================")
        
        # Check if URL is a PDF - if so, download and upload it
        if is_pdf_url(url):
            logger.info("Detected PDF URL - using download/upload strategy")
            return await process_pdf_url(url, model, allowed_nodes, allowed_relationships, enable_communities, extract_bibliographic_info)
        
        # Determine source type based on URL
        if "wikipedia.org" in url:
            source_type = "Wikipedia"
            # Backend needs BOTH source_url AND wiki_query for Wikipedia!
            scan_data = create_form_data(
                uri=NEO4J_URI,
                userName=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE,
                source_url=url,        # Full URL needed here too!
                wiki_query=url,         # And here (backend extracts page title)
                model=model,
                source_type=source_type
            )
        elif "youtube.com" in url or "youtu.be" in url:
            source_type = "youtube"
            scan_data = create_form_data(
                uri=NEO4J_URI,
                userName=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE,
                source_url=url,
                model=model,
                source_type=source_type
            )
        else:
            source_type = "web-url"
            scan_data = create_form_data(
                uri=NEO4J_URI,
                userName=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE,
                source_url=url,
                model=model,
                source_type=source_type
            )
        
        # Step 1: Scan URL and create source node
        logger.info(f"Scanning URL: {url}")
        logger.info(f"=== SENDING TO BACKEND ===")
        logger.info(f"Endpoint: {GRAPH_BUILDER_URL}/url/scan")
        logger.info(f"Data: {scan_data}")
        logger.info(f"wiki_query value: {scan_data.get('wiki_query', 'NOT SET')}")
        logger.info(f"source_type: {scan_data.get('source_type', 'NOT SET')}")
        logger.info(f"=========================")
        
        # Write to a debug file we can easily check
        with open('/tmp/mcp_debug.txt', 'w') as f:
            f.write(f"URL received: {url}\n")
            f.write(f"wiki_query being sent: {scan_data.get('wiki_query', 'NOT SET')}\n")
            f.write(f"Full scan_data: {scan_data}\n")
        
        # DEBUG: Log what we're actually sending
        logger.info(f"=== SENDING TO BACKEND /url/scan ===")
        logger.info(f"URL: {GRAPH_BUILDER_URL}/url/scan")
        logger.info(f"scan_data: {scan_data}")
        logger.info(f"====================================")
        
        # Use 'data' parameter - httpx will encode as application/x-www-form-urlencoded by default
        # FastAPI Form() parameters accept both multipart and form-urlencoded
        scan_response = await http_client.post(
            f"{GRAPH_BUILDER_URL}/url/scan",
            data=scan_data
        )
        scan_response.raise_for_status()
        scan_result = scan_response.json()
        
        if scan_result.get("status") != "Success":
            return json.dumps({
                "status": "Failed",
                "message": f"Failed to scan URL: {scan_result.get('message', 'Unknown error')}"
            })
        
        # Get file name from scan result
        # The backend returns: {"status": "Success", "file_name": [{"fileName": "...", ...}], ...}
        file_info = scan_result.get("file_name", [])
        
        # Debug logging
        logger.info(f"Scan result: {scan_result}")
        logger.info(f"File info: {file_info}")
        
        if not file_info:
            return json.dumps({
                "status": "Failed",
                "message": "No file information returned from URL scan",
                "scan_result": scan_result
            })
        
        # Extract file name from the response
        if isinstance(file_info, list) and len(file_info) > 0:
            file_name = file_info[0].get("fileName")
        elif isinstance(file_info, dict):
            file_name = file_info.get("fileName")
        else:
            return json.dumps({
                "status": "Failed", 
                "message": f"Unexpected file_info format: {file_info}"
            })
        
        if not file_name:
            return json.dumps({
                "status": "Failed",
                "message": "Could not extract fileName from scan result"
            })
        
        # Step 2: Extract knowledge graph
        logger.info(f"Extracting knowledge graph from: {file_name}")
        
        # For Wikipedia, extract language from URL (e.g., "en" from en.wikipedia.org)
        language = None
        if source_type == "Wikipedia":
            if "wikipedia.org" in url:
                # Extract language code from URL (e.g., en.wikipedia.org -> "en")
                parts = url.split(".")
                if len(parts) >= 3 and parts[1] == "wikipedia":
                    language = parts[0].split("//")[-1]  # Get "en" from "https://en"
                else:
                    language = "en"  # Default to English
            else:
                language = "en"  # Default fallback
        
        # Convert None to single space to work with unmodified backend
        # Backend splits allowedNodes without None check. Single space " " splits to [' '],
        # then strip() filters it out → empty list = "allow all"
        nodes_param = allowed_nodes if allowed_nodes else " "
        rels_param = allowed_relationships if allowed_relationships else " "
        
        logger.info(f"Schema params: allowedNodes={repr(nodes_param)}, allowedRelationship={repr(rels_param)}")
        
        extract_data = create_form_data(
            uri=NEO4J_URI,
            userName=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            model=model,
            source_type=source_type,
            file_name=file_name,
            source_url=url if source_type != "Wikipedia" else None,
            # For Wikipedia: send page title (file_name) as wiki_query, not the full URL
            wiki_query=file_name if source_type == "Wikipedia" else None,
            language=language if source_type == "Wikipedia" else None,
            allowedNodes=nodes_param,
            allowedRelationship=rels_param,
            # Required chunking parameters (leaving buffer for punkt tokenizer quirks)
            token_chunk_size=500,   # tokens per chunk (max 512 with buffer)
            chunk_overlap=50,       # token overlap between chunks (10% of chunk_size)
            chunks_to_combine=1     # number of chunks to combine
            # retry_condition defaults to None (first run, not a retry)
        )
        
        logger.info(f"Extract data keys: {list(extract_data.keys())}")
        logger.info(f"allowedNodes in data: {extract_data.get('allowedNodes')}")
        
        logger.info(f"Calling /extract with file_name: {file_name}")
        logger.info(f"Extract data: {extract_data}")
        
        extract_response = await http_client.post(
            f"{GRAPH_BUILDER_URL}/extract",
            data=extract_data
        )
        extract_response.raise_for_status()
        extract_result = extract_response.json()
        
        logger.info(f"Extract response: {extract_result}")
        
        if extract_result.get("status") == "Success":
            data = extract_result.get("data", {})
            response_data = {
                "status": "Success",
                "message": f"Knowledge graph built successfully from {url}",
                "file_name": data.get("fileName"),
                "node_count": data.get("nodeCount", 0),
                "relationship_count": data.get("relationshipCount", 0),
                "processing_time": data.get("total_processing_time", 0),
                "model": model,
                "source_type": source_type
            }
            
            # Include schema information if specified
            if allowed_nodes:
                response_data["schema_nodes"] = allowed_nodes
            if allowed_relationships:
                response_data["schema_relationships"] = allowed_relationships
            if not allowed_nodes and not allowed_relationships:
                response_data["schema_note"] = "No schema restrictions - all entity and relationship types were extracted"
            
            # Optional: Run community detection
            if enable_communities:
                logger.info("Running community detection...")
                try:
                    community_data = create_form_data(
                        uri=NEO4J_URI,
                        userName=NEO4J_USERNAME,
                        password=NEO4J_PASSWORD,
                        database=NEO4J_DATABASE,
                        tasks='["enable_communities"]'  # JSON array with the task
                    )
                    
                    community_response = await http_client.post(
                        f"{GRAPH_BUILDER_URL}/post_processing",
                        data=community_data
                    )
                    community_response.raise_for_status()
                    community_result = community_response.json()
                    
                    if community_result.get("status") == "Success":
                        # Update response with community counts
                        community_counts = community_result.get("data", [])
                        if community_counts:
                            # Find the counts for this file
                            file_counts = next((c for c in community_counts if c.get("filename") == file_name), {})
                            response_data["community_node_count"] = file_counts.get("communityNodeCount", 0)
                            response_data["community_rel_count"] = file_counts.get("communityRelCount", 0)
                        response_data["communities_enabled"] = True
                        logger.info(f"Community detection completed: {response_data.get('community_node_count', 0)} communities")
                    else:
                        logger.warning(f"Community detection failed: {community_result.get('message')}")
                        response_data["communities_enabled"] = False
                        response_data["community_warning"] = "Community detection failed but extraction succeeded"
                except Exception as e:
                    logger.error(f"Error in community detection: {e}")
                    response_data["communities_enabled"] = False
                    response_data["community_warning"] = f"Community detection error: {str(e)}"
            
            return json.dumps(response_data, indent=2)
        else:
            return json.dumps({
                "status": "Failed",
                "message": extract_result.get("message", "Extraction failed"),
                "error": extract_result.get("error")
            })
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error building knowledge graph: {e}")
        return json.dumps({
            "status": "Failed",
            "message": f"HTTP error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}")
        return json.dumps({
            "status": "Failed",
            "message": f"Error: {str(e)}"
        })


@mcp.tool()
async def list_graph_sources() -> str:
    """
    List all sources (documents, URLs, etc.) that have been processed and added to the knowledge graph.
    
    Returns a list of sources with their status, processing information, and metadata.
    
    Returns:
        JSON string with list of sources in the graph
    """
    try:
        data = create_form_data(
            uri=NEO4J_URI,
            userName=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        
        logger.info("Fetching sources list from knowledge graph")
        response = await http_client.post(
            f"{GRAPH_BUILDER_URL}/sources_list",
            data=data
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "Success":
            sources = result.get("data", [])
            return json.dumps({
                "status": "Success",
                "total_sources": len(sources),
                "sources": sources
            }, indent=2)
        else:
            return json.dumps({
                "status": "Failed",
                "message": result.get("message", "Failed to fetch sources"),
                "error": result.get("error")
            })
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error listing sources: {e}")
        return json.dumps({
            "status": "Failed",
            "message": f"HTTP error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        return json.dumps({
            "status": "Failed",
            "message": f"Error: {str(e)}"
        })


@mcp.tool()
async def get_graph_schema() -> str:
    """
    Get the current schema of the knowledge graph including entity types and relationship types.
    
    This shows what kinds of nodes and relationships exist in the graph, which is useful
    for understanding the graph structure and formulating queries.
    
    Returns:
        JSON string with node labels and relationship types in the graph
    """
    try:
        data = create_form_data(
            uri=NEO4J_URI,
            userName=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        
        logger.info("Fetching graph schema")
        response = await http_client.post(
            f"{GRAPH_BUILDER_URL}/schema",
            data=data
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "Success":
            data = result.get("data", {})
            return json.dumps({
                "status": "Success",
                "schema": data.get("triplets", []),
                "message": "Graph schema retrieved successfully"
            }, indent=2)
        else:
            return json.dumps({
                "status": "Failed",
                "message": result.get("message", "Failed to fetch schema"),
                "error": result.get("error")
            })
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error getting schema: {e}")
        return json.dumps({
            "status": "Failed",
            "message": f"HTTP error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        return json.dumps({
            "status": "Failed",
            "message": f"Error: {str(e)}"
        })


if __name__ == "__main__":
    mcp.run()

