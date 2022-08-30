from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor.utils import fetch_archive_from_http
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.eval import EvalAnswers, EvalDocuments
from haystack.reader.farm import FARMReader
from haystack.preprocessor import PreProcessor
from haystack.utils import launch_es
from haystack import Pipeline
from haystack.retriever.sparse import TfidfRetriever
from haystack.modeling.utils import initialize_device_settings
from haystack.document_store import InMemoryDocumentStore
import logging

logger = logging.getLogger(__name__)


def pipeline_evaluation():
    doc_index = "deployqa_docs"
    label_index = "deployqa_labels"

    device, n_gpu = initialize_device_settings(use_cuda=True)

    document_store = ElasticsearchDocumentStore(
        host="localhost", username="", password="", index="document",
        create_index=False, embedding_field="emb",
        embedding_dim=768, excluded_meta_data=["emb"]
    )

    document_store.delete_documents(index=doc_index)
    document_store.delete_documents(index=label_index)

    document_store.add_eval_data(
        filename="../DeQuAD_test.json",
        doc_index=doc_index,
        label_index=label_index,
    )

    labels = document_store.get_all_labels_aggregated(index=label_index)

    'Initialize BM25 TF-IDF DPR Retriever'
    # retriever = DensePassageRetriever(document_store=document_store,
    #                                   query_embedding_model="/mnt/sda/haystack/haystack/test_model/dpr/query_encoder",
    #                                   passage_embedding_model="/mnt/sda/haystack/haystack/test_model/dpr/passage_encoder",
    #                                   use_gpu=True,
    #                                   embed_title=True,
    #                                   top_k=5,
    #                                   )
    # document_store.update_embeddings(retriever, index=doc_index)

    # retriever = TfidfRetriever(document_store=document_store,
    #                            top_k=1
    #                            )

    retriever = ElasticsearchRetriever(document_store=document_store,
                                       top_k=1
                                       )

    retriever_eval_results = retriever.eval(top_k=5, label_index=label_index, doc_index=doc_index, open_domain=True)
    print("Retriever Recall:", retriever_eval_results["recall"])
    print("Retriever Mean Avg Precision:", retriever_eval_results["map"])
    print("Mean Retriever Precision:", retriever_eval_results["mrr"])
    print("n_questions:", retriever_eval_results["n_questions"])
    print("topk:", retriever_eval_results["top_k"])
    print("correct_retrievals:", retriever_eval_results["correct_retrievals"])
    print("number_of_questions:", retriever_eval_results["number_of_questions"])

    reader = FARMReader(
        model_name_or_path="../pretrained_model",
        top_k=3,
        no_ans_boost=0,
        return_no_answer=False,
    )

    eval_retriever = EvalDocuments()
    eval_reader = EvalAnswers(skip_incorrect_retrieval=True, open_domain=True)

    p = Pipeline()
    p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
    p.add_node(component=eval_retriever, name="EvalDocuments", inputs=["ESRetriever"])
    p.add_node(component=reader, name="QAReader", inputs=["EvalDocuments"])
    p.add_node(component=eval_reader, name="EvalAnswers", inputs=["QAReader"])
    results = []

    for l in labels:
        res = p.run(
            query=l.question,
            labels=l,
            params={"index": doc_index, "Retriever": {"top_k": 1}, "Reader": {"top_k": 1}}
        )
        results.append(res)

    eval_retriever.print()
    print()
    eval_reader.print(mode="reader")
    print()
    eval_reader.print(mode="pipeline")
    print()
    retriever.print_time()
    print()
    reader.print_time()


if __name__ == "__main__":
    import torch
    import os

    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pipeline_evaluation()
