# AI-Powered Document Q&A System

A document Q&A system based on LangChain and OpenAI/Zhipu AI/Deepseek, supporting Chinese interaction. It can retrieve relevant information from PDF documents and generate professional answers based on user questions.

## Features

- Supports automatic loading and processing of various document types (pdf, doc, docx, txt, md, csv, xls, xlsx, ppt, pptx, epub)
- Supports multiple embedding models:
  - OpenAI text-embedding-ada-002
  - Zhipu AI embedding-3 (2048 dimensions)
- Supports incremental addition of new documents to avoid duplicate processing
- Supports multiple conversational models:
  - ChatGPT (gpt-3.5-turbo)
  - DeepSeek Chat
- Provides document source tracking
- Full Chinese support

## System Requirements

- Python 3.8+
- OpenAI API key (for using OpenAI models)
- Zhipu AI API key (for using Zhipu AI models)
- DeepSeek API key (for using DeepSeek models)

## Installation Steps

### 1. Clone the project and enter the directory

```bash
git clone <git url>
cd <directory>
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate # MacOS
venv/Scripts/activate # Windows
# On Windows, you need to install VS C++ first
# On Windows, to process Word files, you also need to install pandoc: https://github.com/jgm/pandoc/releases
# On MacOS, to process doc files, install: brew install --cask libreoffice
pip install -r requirements.txt
```

### 3. Configure environment variables

- Refer to `.env.sample` to create a `.env` file
- Add the corresponding API keys as needed:

```bash
# OpenAI (optional)
OPENAI_API_KEY=your-openai-api-key-here

# Zhipu AI (optional)
ZHIPUAI_API_KEY=your-zhipuai-api-key-here

# DeepSeek (optional)
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

## Usage

### 1. Prepare Documents

Place the PDF documents to be processed in the `data` directory (or another specified directory). Subdirectories are supported.

### 2. Create Vector Database

Run the `create_embeddings.py` script to process documents and create the vector database:

```bash
# Use default OpenAI embeddings
python create_embeddings.py

# Use Zhipu AI embeddings
python create_embeddings.py --chromadir chroma_db_zhipuai

# Or specify a custom directory
python create_embeddings.py --datadir /path/to/pdf/files
```

Features:

- Automatically detects and processes all supported files in the directory
- Supports incremental updates, only processing newly added documents
- Retains existing vector database content
- Automatically selects the embedding model based on the database directory name

### 3. Start the Q&A System

Run `qa_interface.py` to start the Q&A interaction:

```bash
# Use the default OpenAI model
python qa_interface.py

# Use the DeepSeek model
python qa_interface.py --model deepseek

# Use Zhipu AI embeddings (by specifying the database directory)
python qa_interface.py --chromadir chroma_db_zhipuai
```

Instructions:

- Enter your question and press Enter
- Type 'exit' or 'quit' to exit
- Type 'help' for help

Supported models:

- OpenAI (default): Uses the gpt-3.5-turbo model
- DeepSeek: Uses the deepseek-chat model
- Zhipu AI: Uses the embedding-3 model for text embeddings

Note: Different models require the corresponding API key:

- OpenAI model requires the `OPENAI_API_KEY` environment variable
- DeepSeek model requires the `DEEPSEEK_API_KEY` environment variable
- Zhipu AI requires the `ZHIPUAI_API_KEY` environment variable

### Example Q&A

```txt
🤖 Hardware Document Q&A System
- Enter your question and press Enter
- Type 'exit' or 'quit' to exit
- Type 'help' for help

> What is the LED drive function of RL78/F13?

🧠 Answer:
The LED drive function of RL78/F13 includes an 8-bit D/A converter and a built-in comparator. This chip supports a power supply voltage of 2.7 to 5.5V and operates normally at ambient temperatures from -40 to +105°C. In addition, RL78/F13 has LIN integration, suitable for general automotive electrical applications (such as motor control, door control, lighting control, etc.) and motorcycle engine control.

Document: RL78/F13, F14
Section: CHAPTER 1 OVERVIEW
Pages: 8, 9

📄 Reference:
 - data/r01uh0368ej0220_rl78f13_hardware.pdf
```

## Project Structure

```dir
.
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
├── .env                   # Environment variable configuration
├── data/                  # PDF document directory
├── create_embeddings.py   # Document processing and vectorization script
├── qa_interface.py        # Q&A interaction interface
├── embeddings.py          # Embedding model implementation
└── chroma_db/            # Vector database storage directory
```

## Notes

1. API keys must be configured for the first run
2. PDF documents must be in a copyable text format
3. The vector database will occupy some disk space
4. It is recommended to back up the vector database directory regularly
5. Zhipu AI embeddings use 2048-dimensional vectors, which may require more storage space

## FAQ

1. **File not found**
   - Make sure the file is in the correct directory
   - Check file permissions

2. **API key error**
   - Check if the key in the `.env` file is correct
   - Make sure the API key has not expired
   - Ensure the correct environment variable name is used

3. **High memory usage**
   - You can adjust the `CHUNK_SIZE` parameter
   - Process large documents in batches
   - Note that Zhipu AI embeddings have a larger vector dimension

## Recommended: ZhipuAI + Deepseek for use in China

```shell
# Build document index
python create_embeddings.py --model zhipuai --chromadir chroma_db_zhipuai
# Query
python qa_interface.py --chromadir chroma_db_zhipuai --model deepseek
```

## Web Page

```shell
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

```shell
# Build docker image
docker buildx build --platform linux/amd64 --push -t registry.cn-shanghai.aliyuncs.com/zhitek/rag-qa-app:latest .
```

## Incremental embedding

1. Put new files into data directory
2. Run create_embeddings.py script
3. Upload the updated chroma db files to the server
4. Restart rag service
5. Make sure the new db files are mounted into the container
