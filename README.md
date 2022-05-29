# DeployQA
Deploy QA is a tool that can automatically answers software deployment questions over user manuals and Stack Overflow posts.
![system overview](pic/mainpage.png) 


## Architecture
![ui](pic/architecture.png) 


The overall architecture of DeployQA is shown here. It leverages a retrieval-and-reader Framework. 
Given a question, a retriever first searches for candidate documents from a collection of user manuals and Stack Overflow posts. Then, a reader predicts the answer span from the selected documents using a domain-adapted RoBERTa model.

## Install
You can use docker to directly deploy our tool.

**1. Step into DeployQA**
```
cd DeployQA
```

**2. Pull docker images**
```
docker-compose pull
```

**3. Launch containers**
```
docker-compose up
```

**Note**: The following containers listens three ports:
* DeployQA Framework: listens on port 8000
* Elasticsearch: listens on port 9200
* Streamlit UI: listens on port 8501


## Evaluation
We evaluate the effectiveness of the proposed DeployQA on DeQuAD, a dataset that we created with 2,000 QAs from Stack Overflow. Experimental results show that DeployQA achieves
an F1-score of 49.85%, which significantly outperforms state-of-the-art approaches.

<img src="https://github.com/Smallqqqq/DeployQA/blob/main/pic/evaluation.png" width="660" height="239">


