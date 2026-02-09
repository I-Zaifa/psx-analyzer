"""
PSX Analyzer – Main Entry Point
Usage:
    python run.py --pipeline              # Run data pipeline only
    python run.py --server                # Start web server only
    python run.py                         # Run pipeline then start server
    python run.py --pipeline --max 10     # Test with 10 symbols
    python run.py --pipeline --skip-fund  # Skip fundamentals scraping
"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="PSX Analyzer - Pakistan Stock Exchange Analysis Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    Full pipeline + start server
  python run.py --pipeline         Run data pipeline only
  python run.py --server           Start web dashboard only
  python run.py --pipeline --max 5 Test with 5 symbols
  python run.py --port 8080        Use custom port
        """
    )
    parser.add_argument("--pipeline", action="store_true",
                        help="Run data pipeline only (no server)")
    parser.add_argument("--server", action="store_true",
                        help="Start web server only (no pipeline)")
    parser.add_argument("--skip-fund", action="store_true",
                        help="Skip fundamentals scraping (faster)")
    parser.add_argument("--max", type=int, default=None,
                        help="Max symbols to process (for testing)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Web server port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Web server host (default: 127.0.0.1)")
    parser.add_argument("--icons", action="store_true",
                        help="Generate PWA icon PNGs")

    args = parser.parse_args()

    # Generate icons if requested
    if args.icons:
        from generate_icons import create_png
        icons_dir = os.path.join(os.path.dirname(__file__), "app", "static", "icons")
        os.makedirs(icons_dir, exist_ok=True)
        create_png(192, 192, os.path.join(icons_dir, "icon-192.png"))
        create_png(512, 512, os.path.join(icons_dir, "icon-512.png"))
        print("PWA icons generated.")
        if not args.pipeline and not args.server:
            return

    # Default: run both pipeline and server
    run_pipeline = not args.server or args.pipeline
    run_server = not args.pipeline or args.server

    # If neither flag is set, run both
    if not args.pipeline and not args.server:
        run_pipeline = True
        run_server = True

    if run_pipeline:
        print("\n" + "=" * 60)
        print("  PSX ANALYZER - DATA PIPELINE")
        print("=" * 60 + "\n")

        from backend.run_pipeline import run_pipeline as execute_pipeline
        success = execute_pipeline(
            skip_fundamentals=args.skip_fund,
            max_symbols=args.max
        )

        if not success:
            print("\nPipeline failed. Check logs in data/logs/")
            if not run_server:
                sys.exit(1)
            print("Starting server anyway with whatever data is available...\n")

    if run_server:
        print("\n" + "=" * 60)
        print(f"  PSX ANALYZER - WEB DASHBOARD")
        print(f"  http://{args.host}:{args.port}")
        print("=" * 60 + "\n")

        from app.app import app
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
