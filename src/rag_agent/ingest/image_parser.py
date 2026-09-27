"""Parser for raster images (.png, .jpg): the file and one image node, described after indexing by the
image pipeline (ТЗ S4, Э5). Until then the node has no text and is not embedded."""

from __future__ import annotations

from pathlib import Path

from rag_agent.config import ChunkingConfig
from rag_agent.ingest.walker import CorpusFile
from rag_agent.schema import Edge, EdgeType, FileType, Node, NodeType, ParsedFile


def parse_image(cf: CorpusFile, cfg: ChunkingConfig) -> ParsedFile:
    rel = cf.rel_path
    result = ParsedFile(file_path=rel, file_type=FileType.PNG)
    name = Path(rel).name
    file_node = Node(id=rel, file_path=rel, file_type=FileType.PNG, node_type=NodeType.FILE, title=name,
                     metadata={"size": cf.size}, embed=False)
    image = Node(id=f"{rel}#image", file_path=rel, file_type=FileType.PNG, node_type=NodeType.IMAGE,
                 parent_id=file_node.id, title=name, text="", metadata={"image_source": "file"}, embed=False)
    result.nodes += [file_node, image]
    result.edges.append(Edge(src=file_node.id, dst=image.id, type=EdgeType.CONTAINS))
    return result
