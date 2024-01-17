# Query System

## Overview

Welcome to the Query System project, a powerful tool for information retrieval based on a Boolean retrieval model coupled with a Vector Space model. This versatile system allows users to perform phrase queries, retrieve documents ranked by similarity scores, and gain insights from processed text data.

## Features

- **Boolean Retrieval Model**: A robust Boolean retrieval model forms the foundation of the system, enabling users to execute precise and complex queries using operators such as "AND," "OR," and "NOT."

- **Vector Space Model**: Leveraging a Vector Space model enhances the system's capability to retrieve documents ranked based on similarity scores, providing users with relevant and ordered results.

- **Phrase Queries**: The system supports phrase queries, allowing users to search for specific phrases within the document collection.

- **Internal Workflow**: A streamlined internal workflow processes the collection of text data, employs Natural Language Processing (NLP) techniques, and generates basic insights. The workflow also constructs a positional index of the collection for efficient query processing.

## Project Structure

The code is organized into two main parts:

1. **Query-System-notebooks**: This section contains Jupyter Notebooks explaining the underlying logic, implementation details, and examples of using the Query System.

2. **Query-System-CMD**: Find the main executable file for running the Query System via the command line. Refer to this file to interact with the system using the command-line interface.

## Usage

Explore the `Query-System-notebooks` Jupyter Notebooks file for in-depth explanations and examples of using the Query System.

## Getting Started

To run the Query System on your collection of files, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the `Query-System-CMD` directory.
3. Open the `Query-System-CMD` python file and change the commented line of code to your collection directory that you want to run the query system on
4. Run the main executable file to start the system in the command-line interface.
