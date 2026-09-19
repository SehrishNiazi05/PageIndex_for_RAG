import argparse
import json
from .audit import run as audit
from .store import ingest


def main():
    parser = argparse.ArgumentParser(description="AI Dentist local pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    p = sub.add_parser("ingest"); p.add_argument("--book"); p.add_argument("--no-dense", action="store_true"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("retrieve"); p.add_argument("question"); p.add_argument("--book")
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("--audience", default="clinician", choices=["clinician", "patient"])
    p = sub.add_parser("serve"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("eval"); p.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        report = audit(); print(json.dumps({"books":len(report["books"]), "records":sum(b["records"] for b in report["books"])}, indent=2))
    elif args.command == "ingest": print(json.dumps(ingest(args.book, not args.no_dense, args.force), indent=2))
    elif args.command == "retrieve":
        from .retrieve import Retriever
        hits, stages = Retriever().search(args.question, args.book)
        print(json.dumps({"hits":hits,"stages":stages}, indent=2, ensure_ascii=False))
    elif args.command == "ask":
        from .agent import ask
        print(json.dumps(ask(args.question, args.audience), indent=2, ensure_ascii=False))
    elif args.command == "serve":
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            parser.error("The v1 app serves localhost only")
        import uvicorn
        uvicorn.run("rag.app:app", host=args.host, port=args.port)
    elif args.command == "eval":
        from .eval.run import run
        print(json.dumps(run(generate=args.generate), indent=2))


if __name__ == "__main__": main()
