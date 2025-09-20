#!/usr/bin/env python3
"""
AgentCortex CLI Interface

Main entry point for the AgentCortex RAG pipeline.
Provides command-line interface for common operations.
"""

import argparse
import logging
import sys
from pathlib import Path

from agentcortex.utils import setup_logging, Config
from agentcortex.retrieval import RAGRetriever
from agentcortex.pdf_extraction import PDFExtractor
from agentcortex.toxicity_filter import ToxicityFilter
from agentcortex.langextract import LanguageExtractor


def setup_cli_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="AgentCortex: Comprehensive RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract text from PDF
  python main.py extract-pdf document.pdf --output text.txt

  # Create RAG index from documents
  python main.py create-index docs/ --output-dir ./rag_index

  # Query RAG system
  python main.py query "What is machine learning?" --index ./rag_index

  # Filter toxic content
  python main.py filter-toxicity input.txt --output clean.txt

  # Run complete pipeline
  python main.py pipeline --config config.yaml --input docs/ --output ./pipeline
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Extract PDF command
    pdf_parser = subparsers.add_parser("extract-pdf", help="Extract text from PDF")
    pdf_parser.add_argument("input", help="Input PDF file")
    pdf_parser.add_argument("--output", "-o", help="Output text file")
    pdf_parser.add_argument("--preserve-layout", action="store_true", 
                           help="Preserve text layout")
    
    # Create index command
    index_parser = subparsers.add_parser("create-index", help="Create RAG index")
    index_parser.add_argument("input", help="Input directory or file")
    index_parser.add_argument("--output-dir", "-o", required=True,
                             help="Output directory for index")
    index_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                             help="Embedding model name")
    index_parser.add_argument("--chunk-size", type=int, default=1000,
                             help="Text chunk size")
    index_parser.add_argument("--chunk-overlap", type=int, default=200,
                             help="Chunk overlap size")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query RAG system")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--index", required=True, help="Path to RAG index")
    query_parser.add_argument("--top-k", type=int, default=5,
                             help="Number of results to return")
    query_parser.add_argument("--score-threshold", type=float,
                             help="Minimum similarity score")
    
    # Filter toxicity command
    filter_parser = subparsers.add_parser("filter-toxicity", 
                                         help="Filter toxic content")
    filter_parser.add_argument("input", help="Input text file")
    filter_parser.add_argument("--output", "-o", help="Output file")
    filter_parser.add_argument("--threshold", type=float, default=0.7,
                              help="Toxicity threshold")
    filter_parser.add_argument("--remove", action="store_true",
                              help="Remove toxic content instead of cleaning")
    
    # Complete pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", 
                                           help="Run complete RAG pipeline")
    pipeline_parser.add_argument("--input", required=True,
                                help="Input directory or file")
    pipeline_parser.add_argument("--output", required=True,
                                help="Output directory")
    pipeline_parser.add_argument("--filter-toxicity", action="store_true",
                                help="Enable toxicity filtering")
    pipeline_parser.add_argument("--filter-language", 
                                help="Filter by language (e.g., 'en')")
    
    return parser


def extract_pdf_command(args, config):
    """Handle PDF extraction command."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting text from PDF: {args.input}")
    
    extractor = PDFExtractor(preserve_layout=args.preserve_layout)
    
    try:
        text = extractor.extract_text(args.input)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Text saved to: {args.output}")
        else:
            print(text)
        
        # Show PDF info
        pdf_info = extractor.get_pdf_info(args.input)
        logger.info(f"PDF info: {pdf_info}")
        
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}")
        return 1
    
    return 0


def create_index_command(args, config):
    """Handle index creation command."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating RAG index from: {args.input}")
    
    retriever = RAGRetriever(
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            # Single file
            if input_path.suffix.lower() == '.pdf':
                retriever.add_pdf_documents([str(input_path)])
            else:
                # Text file
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                retriever.add_documents_from_text([text])
        
        elif input_path.is_dir():
            # Directory of files
            pdf_files = list(input_path.glob("**/*.pdf"))
            txt_files = list(input_path.glob("**/*.txt"))
            
            if pdf_files:
                logger.info(f"Processing {len(pdf_files)} PDF files")
                retriever.add_pdf_documents([str(f) for f in pdf_files])
            
            if txt_files:
                logger.info(f"Processing {len(txt_files)} text files")
                texts = []
                for txt_file in txt_files:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                retriever.add_documents_from_text(texts)
        
        else:
            logger.error(f"Input path not found: {args.input}")
            return 1
        
        # Save index
        retriever.save(args.output_dir)
        logger.info(f"Index saved to: {args.output_dir}")
        
        # Show statistics
        stats = retriever.get_retriever_stats()
        logger.info(f"Index statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        return 1
    
    return 0


def query_command(args, config):
    """Handle query command."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Querying RAG system: {args.query}")
    
    try:
        retriever = RAGRetriever()
        retriever.load(args.index)
        
        results = retriever.retrieve(
            query=args.query,
            k=args.top_k,
            score_threshold=args.score_threshold
        )
        
        logger.info(f"Found {len(results)} relevant documents:")
        
        for i, doc in enumerate(results, 1):
            score = doc.metadata.get('score', 0.0)
            source = doc.metadata.get('source_file', 'Unknown')
            
            print(f"\n--- Result {i} ---")
            print(f"Score: {score:.3f}")
            print(f"Source: {source}")
            print(f"Content:\n{doc.page_content}")
        
    except Exception as e:
        logger.error(f"Failed to query system: {e}")
        return 1
    
    return 0


def filter_toxicity_command(args, config):
    """Handle toxicity filtering command."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Filtering toxicity from: {args.input}")
    
    try:
        # Read input file
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        toxicity_filter = ToxicityFilter(
            toxicity_threshold=args.threshold,
            use_rule_based=True
        )
        
        # Filter content
        clean_lines, analysis = toxicity_filter.filter_dataset(
            lines,
            remove_toxic=args.remove,
            clean_toxic=not args.remove
        )
        
        # Save results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.writelines(clean_lines)
            logger.info(f"Clean text saved to: {args.output}")
        else:
            for line in clean_lines:
                print(line.strip())
        
        # Show statistics
        stats = toxicity_filter.get_statistics(analysis)
        logger.info(f"Filtering results: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to filter toxicity: {e}")
        return 1
    
    return 0


def pipeline_command(args, config):
    """Handle complete pipeline command."""
    logger = logging.getLogger(__name__)
    
    logger.info("Running complete RAG pipeline")
    
    try:
        # Initialize components
        retriever = RAGRetriever()
        
        if args.filter_toxicity:
            toxicity_filter = ToxicityFilter()
        
        if args.filter_language:
            lang_extractor = LanguageExtractor()
        
        # Process input
        input_path = Path(args.input)
        texts = []
        metadata = []
        
        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                retriever.add_pdf_documents([str(input_path)])
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    texts = [f.read()]
                    metadata = [{"source": str(input_path)}]
        
        elif input_path.is_dir():
            # Process directory
            for file_path in input_path.rglob("*"):
                if file_path.is_file():
                    if file_path.suffix.lower() == '.pdf':
                        retriever.add_pdf_documents([str(file_path)])
                    elif file_path.suffix.lower() in ['.txt', '.md']:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                            metadata.append({"source": str(file_path)})
        
        # Apply filters if requested
        if texts:
            if args.filter_language:
                logger.info(f"Filtering by language: {args.filter_language}")
                filtered_texts, indices, lang_results = lang_extractor.filter_by_language(
                    texts, [args.filter_language], min_confidence=0.8
                )
                texts = filtered_texts
                metadata = [metadata[i] for i in indices]
            
            if args.filter_toxicity:
                logger.info("Applying toxicity filtering")
                clean_texts, toxicity_analysis = toxicity_filter.filter_dataset(
                    texts, clean_toxic=True
                )
                texts = clean_texts
            
            # Add to retriever
            retriever.add_documents_from_text(texts, metadata)
        
        # Save pipeline
        retriever.save(args.output)
        logger.info(f"Pipeline saved to: {args.output}")
        
        # Show statistics
        stats = retriever.get_retriever_stats()
        logger.info(f"Pipeline statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(
        level=log_level,
        log_file=args.log_file,
        console_output=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AgentCortex CLI")
    
    # Load configuration
    config = Config(args.config if args.config else None)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error("Configuration validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return 1
    
    # Execute command
    if args.command == "extract-pdf":
        return extract_pdf_command(args, config)
    
    elif args.command == "create-index":
        return create_index_command(args, config)
    
    elif args.command == "query":
        return query_command(args, config)
    
    elif args.command == "filter-toxicity":
        return filter_toxicity_command(args, config)
    
    elif args.command == "pipeline":
        return pipeline_command(args, config)
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
