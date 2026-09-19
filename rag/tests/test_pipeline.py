import os
import unittest
from unittest.mock import patch
from rag.agent import ask
from rag.chunker import chunk_book
from rag.config import PROCESSED, MAX_TOKENS


class FakeRetriever:
    def search(self, query):
        return ([{"chunk_id":"abc", "book":"Example", "chapter":"Chapter 1", "page_start":12,
                  "source_ids":["Example#1"], "text":"The tooth is protected by enamel.", "n_tokens":20}],
                {"selected":["abc"]})


class PipelineTests(unittest.TestCase):
    def test_chunks_are_bounded_and_have_source_and_page(self):
        path = PROCESSED / "Oral Medicine and Radiology.jsonl"
        chunks = list(chunk_book(path))
        self.assertTrue(chunks)
        self.assertTrue(all(c["n_tokens"] <= MAX_TOKENS and c["source_ids"] and c["page_start"] for c in chunks))

    def test_patient_personal_question_does_not_retrieve(self):
        with patch("rag.agent.get_retriever", side_effect=AssertionError("retrieval was called")):
            result = ask("My gums are bleeding; what medicine should I take?", "patient")
        self.assertEqual(result["status"], "personal_question")
        self.assertEqual(result["citations"], [])

    def test_fabricated_citation_is_rejected(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY":"test"}), patch("rag.agent._llm", side_effect=[
            {"sufficient":True}, {"claims":[{"text":"Enamel protects teeth.","evidence_id":"E99","quote":"The tooth is protected by enamel."}]},
            {"claims":[]}]):
            result = ask("What protects teeth?", retriever=FakeRetriever())
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["citations"], [])

    def test_quote_must_appear_in_source(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY":"test"}), patch("rag.agent._llm", side_effect=[
            {"sufficient":True}, {"claims":[{"text":"Enamel protects teeth.","evidence_id":"E1","quote":"Enamel cures all dental disease."}]},
            {"claims":[]}]):
            result = ask("What protects teeth?", retriever=FakeRetriever())
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_citation_uses_quoted_records_page(self):
        class TwoPageRetriever:
            def search(self, query):
                return ([{"chunk_id":"two", "book":"Example", "chapter":"Chapter 1", "page_start":12,
                          "source_ids":["Example#1", "Example#2"], "source_spans":[
                              {"source_id":"Example#1","page":12,"start":0,"end":13},
                              {"source_id":"Example#2","page":13,"start":15,"end":52}],
                          "text":"First passage\n\nThe tooth is protected by enamel.", "n_tokens":30}],
                        {"selected":["two"]})
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY":"test"}), patch("rag.agent._llm", side_effect=[
            {"sufficient":True},
            {"claims":[{"text":"Enamel protects teeth.","evidence_id":"E1","quote":"The tooth is protected by enamel."}]},
            {"supported":[True],"answers_question":True,"patient_specific_advice":False}]):
            result = ask("What protects teeth?", retriever=TwoPageRetriever())
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"][0]["pdf_page"], 13)
        self.assertEqual(result["citations"][0]["source_id"], "Example#2")

    def test_unsupported_extra_claim_does_not_discard_supported_answer(self):
        class OnePageRetriever:
            def search(self, query):
                text = "The tooth is protected by enamel. Enamel is a hard outer layer."
                return ([{"chunk_id":"one", "book":"Example", "chapter":"Chapter 1", "page_start":12,
                          "source_ids":["Example#1"], "source_spans":[
                              {"source_id":"Example#1","page":12,"start":0,"end":len(text)}],
                          "text":text, "n_tokens":20}], {"selected":["one"]})
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY":"test"}), patch("rag.agent._llm", side_effect=[
            {"sufficient":True},
            {"claims":[{"text":"Enamel protects teeth.","evidence_id":"E1","quote":"The tooth is protected by enamel."},
                       {"text":"Enamel cures all dental disease.","evidence_id":"E1","quote":"Enamel is a hard outer layer."}]},
            {"supported":[True,False],"answers_question":True,"patient_specific_advice":False}]):
            result = ask("What protects teeth?", retriever=OnePageRetriever())
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "Enamel protects teeth. [E1]")
        self.assertEqual(len(result["citations"]), 1)

    def test_supported_claims_must_answer_question(self):
        class OnePageRetriever:
            def search(self, query):
                text = "The tooth is protected by enamel."
                return ([{"chunk_id":"one", "book":"Example", "chapter":"Chapter 1", "page_start":12,
                          "source_ids":["Example#1"], "source_spans":[
                              {"source_id":"Example#1","page":12,"start":0,"end":len(text)}],
                          "text":text, "n_tokens":20}], {"selected":["one"]})
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY":"test"}), patch("rag.agent._llm", side_effect=[
            {"sufficient":True},
            {"claims":[{"text":"Enamel protects teeth.","evidence_id":"E1","quote":"The tooth is protected by enamel."}]},
            {"supported":[True],"answers_question":False,"patient_specific_advice":False},
            {"claims":[]}]):
            result = ask("What causes cavities?", retriever=OnePageRetriever())
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["citations"], [])

    def test_retries_a_draft_without_valid_quotes(self):
        class OnePageRetriever:
            def search(self, query):
                text = "The tooth is protected by enamel."
                return ([{"chunk_id":"one", "book":"Example", "chapter":"Chapter 1", "page_start":12,
                          "source_ids":["Example#1"], "source_spans":[
                              {"source_id":"Example#1","page":12,"start":0,"end":len(text)}],
                          "text":text, "n_tokens":20}], {"selected":["one"]})
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY":"test"}), patch("rag.agent._llm", side_effect=[
            {"sufficient":True},
            {"claims":[{"text":"Enamel protects teeth.","evidence_id":"E1","quote":"A fabricated quote."}]},
            {"claims":[{"text":"Enamel protects teeth.","evidence_id":"E1","quote":"The tooth is protected by enamel."}]},
            {"supported":[True],"answers_question":True,"patient_specific_advice":False}]):
            result = ask("What protects teeth?", retriever=OnePageRetriever())
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(result["citations"]), 1)

    def test_each_claim_gets_its_own_quote_citation(self):
        class OnePageRetriever:
            def search(self, query):
                text = "The tooth is protected by enamel. Enamel is a hard outer layer."
                return ([{"chunk_id":"one", "book":"Example", "chapter":"Chapter 1", "page_start":12,
                          "source_ids":["Example#1"], "source_spans":[
                              {"source_id":"Example#1","page":12,"start":0,"end":len(text)}],
                          "text":text, "n_tokens":20}], {"selected":["one"]})
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY":"test"}), patch("rag.agent._llm", side_effect=[
            {"sufficient":True},
            {"claims":[{"text":"Enamel protects teeth.","evidence_id":"E1","quote":"The tooth is protected by enamel."},
                       {"text":"Enamel is the outer layer.","evidence_id":"E1","quote":"Enamel is a hard outer layer."}]},
            {"supported":[True,True],"answers_question":True,"patient_specific_advice":False}]):
            result = ask("What is enamel?", retriever=OnePageRetriever())
        self.assertEqual(result["status"], "answered")
        self.assertIn("Enamel protects teeth. [E1]", result["answer"])
        self.assertIn("Enamel is the outer layer. [E2]", result["answer"])
        self.assertEqual([c["excerpt"] for c in result["citations"]],
                         ["The tooth is protected by enamel.", "Enamel is a hard outer layer."])


if __name__ == "__main__": unittest.main()
