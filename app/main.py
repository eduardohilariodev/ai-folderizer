import json
import os
import shutil
import time
from uuid import uuid4

import faiss
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptWithTemplates
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, model_validator

load_dotenv()

categories = set()


class Entry(BaseModel):
    """Entry representing something, with a name, a category and a type."""

    name: str = Field(description="The name of the entry")
    category: str = Field(description="The category of the entry")
    type: str = Field(description="The type of the entry")
    valid_categories: set[str] = Field(default_factory=set)

    # You can add custom validation logic easily with Pydantic.
    @model_validator(mode="before")
    @classmethod
    def category_is_correct(cls, values) -> dict:
        if not isinstance(values, dict):
            return values
        if values.get("category") not in values.get("valid_categories", set()):
            raise ValueError("Category not found in allowed categories!")
        return values


class File(BaseModel):
    """System file."""

    name: str = Field(description="The name without extension")
    extension: str = Field(description="The extension of the file")
    filename: str = Field(description="The filename with extension")
    path: str = Field(description="The absolute path of the file")


def load_categories(file_path: str) -> set:
    """Load categories from a JSON file and return them as a set.

    Args:
        file_path (str): The path to the JSON file containing categories.

    Returns:
        Set: A set of categories.
    """
    with open(file_path, "r") as file:
        categories = set(json.load(file))

    return categories


def read_files_from_path(path: str, extension: str | None = None) -> list[File]:
    """Read all files from a directory path, optionally filtering by extension.

    Args:
        path (str): Directory path to read files from
        extension (str | None): Optional file extension to filter by (e.g. '.txt'). Defaults to None.

    Returns:
        list[str]: List of file paths found in the directory
    """
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if extension is None or filename.endswith(extension):
                name, _ = os.path.splitext(filename)
                file = File(
                    name=name,
                    extension=extension,
                    filename=filename,
                    path=os.path.join(root, filename),
                )
                files.append(file)
    return files


async def categorize(entries: list[File]) -> list[Entry]:
    """Categorize a list of entries using LLM.

    Args:
        entries: List of dictionaries containing entry info

    Returns:
        list[Entry]: List of categorized entries
    """
    examples = [
        {
            "type": "book",
            "entry": "Becoming by Michelle Obama",
            "output": {
                "name": "Becoming",
                "category": "Biography & Memoir",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Deep Work by Cal Newport",
            "output": {
                "name": "Deep Work",
                "category": "Career & Success",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Crucial Conversations by Kerry Patterson",
            "output": {
                "name": "Crucial Conversations",
                "category": "Communication Skills",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "The Culture Code by Daniel Coyle",
            "output": {
                "name": "The Culture Code",
                "category": "Corporate Culture",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Big Magic by Elizabeth Gilbert",
            "output": {"name": "Big Magic", "category": "Creativity", "type": "book"},
        },
        {
            "type": "book",
            "entry": "Capital in the Twenty-First Century by Thomas Piketty",
            "output": {
                "name": "Capital in the Twenty-First Century",
                "category": "Economics",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "How Children Learn by John Holt",
            "output": {
                "name": "How Children Learn",
                "category": "Education",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Zero to One by Peter Thiel",
            "output": {
                "name": "Zero to One",
                "category": "Entrepreneurship",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "The Great Gatsby by F. Scott Fitzgerald",
            "output": {
                "name": "The Great Gatsby",
                "category": "Fiction",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "In Defense of Food by Michael Pollan",
            "output": {
                "name": "In Defense of Food",
                "category": "Health & Nutrition",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Sapiens by Yuval Noah Harari",
            "output": {"name": "Sapiens", "category": "History", "type": "book"},
        },
        {
            "type": "book",
            "entry": "Good to Great by Jim Collins",
            "output": {
                "name": "Good to Great",
                "category": "Management & Leadership",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Building a StoryBrand by Donald Miller",
            "output": {
                "name": "Building a StoryBrand",
                "category": "Marketing & Sales",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "The Power of Now by Eckhart Tolle",
            "output": {
                "name": "The Power of Now",
                "category": "Mindfulness & Happiness",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Rich Dad Poor Dad by Robert Kiyosaki",
            "output": {
                "name": "Rich Dad Poor Dad",
                "category": "Money & Investments",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "The Alchemist by Paulo Coelho",
            "output": {
                "name": "The Alchemist",
                "category": "Motivation & Inspiration",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Silent Spring by Rachel Carson",
            "output": {
                "name": "Silent Spring",
                "category": "Nature & the Environment",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "How to Talk So Kids Will Listen by Adele Faber",
            "output": {
                "name": "How to Talk So Kids Will Listen",
                "category": "Parenting",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Atomic Habits by James Clear",
            "output": {
                "name": "Atomic Habits",
                "category": "Personal Development",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "The Republic by Plato",
            "output": {
                "name": "The Republic",
                "category": "Philosophy",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "The Prince by Niccolò Machiavelli",
            "output": {"name": "The Prince", "category": "Politics", "type": "book"},
        },
        {
            "type": "book",
            "entry": "Getting Things Done by David Allen",
            "output": {
                "name": "Getting Things Done",
                "category": "Productivity",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Thinking, Fast and Slow by Daniel Kahneman",
            "output": {
                "name": "Thinking, Fast and Slow",
                "category": "Psychology",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "The Power of Myth by Joseph Campbell",
            "output": {
                "name": "The Power of Myth",
                "category": "Religion & Spirituality",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "A Brief History of Time by Stephen Hawking",
            "output": {
                "name": "A Brief History of Time",
                "category": "Science",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Men Are from Mars, Women Are from Venus by John Gray",
            "output": {
                "name": "Men Are from Mars, Women Are from Venus",
                "category": "Sex & Relationships",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "Guns, Germs, and Steel by Jared Diamond",
            "output": {
                "name": "Guns, Germs, and Steel",
                "category": "Society & Culture",
                "type": "book",
            },
        },
        {
            "type": "book",
            "entry": "The Singularity Is Near by Ray Kurzweil",
            "output": {
                "name": "The Singularity Is Near",
                "category": "Technology & the Future",
                "type": "book",
            },
        },
    ]
    to_vectorize = [" ".join(example.values()) for example in examples]
    embeddings = OpenAIEmbeddings()

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    uuids = [str(uuid4()) for _ in range(len(examples))]
    vector_store.add_documents(documents=examples, ids=uuids)

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vector_store,
        k=2,
        input_keys=["entry"],
    )

    # Initialize selector with examples
    example_selector.add_examples(examples)

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    categories = load_categories("app/categories.json")
    categories_str = ", ".join(categories)

    parser = PydanticOutputParser(
        pydantic_object=Entry,
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create the prompt and chain
    prompt = FewShotPromptWithTemplates(
        examples=examples,  # Add examples here
        example_prompt=example_prompt,
        template="""
        You must ONLY select from these exact {type} categories: {categories_str}
        For the {type} titled "{entry}", return ONLY ONE category from the list above that best matches.
        DO NOT create new categories or modify existing ones.
        If no category fits perfectly, choose the closest match from the provided list.
        {format_instructions}
        """,
        example_selector=example_selector,  # Add selector here
        input_variables=["categories_str", "entry", "type"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the chain and add retry functionality
    prompt_and_model = (prompt | llm).with_retry(
        stop_after_attempt=3,  # Number of retries
        wait_exponential_jitter=True,  # Add jitter between retries
        # Specify which exceptions to retry on
        retry_if_exception_type=(OutputParserException, ValueError),
    )

    output = await prompt_and_model.abatch(
        inputs=[
            {
                "categories_str": categories_str,
                "entry": entry,
                "type": "book",
                "valid_categories": categories,
            }
            for entry in entries
        ],
        config=None,
        return_exceptions=True,
    )

    return await parser.abatch(output)


def trim_files_whitespaces(path: str, extension: str) -> None:
    """Remove trailing whitespaces from filenames in the given path.

    Args:
        path: Directory path to process files in
        extension: File extension to filter by
    """
    for filename in os.listdir(path):
        if filename.endswith(extension):
            name, ext = os.path.splitext(filename)
            new_name = name.rstrip() + ext

            if new_name != filename:
                old_path = os.path.join(path, filename)
                new_path = os.path.join(path, new_name)
                os.rename(old_path, new_path)


async def main(path: str = "", extension: str = "", help: bool = False) -> None:
    """Main function to load and display categories.

    Args:
        path: Directory path containing files to categorize
        extension: File extension to filter by
        help: Show help message and exit
    """
    if help or not path or not extension:
        print("""
AI Folderizer - Automatically categorize files into folders

Usage: python app/main.py --path PATH --extension EXTENSION [--help]

Options:
  --path PATH       Directory path containing files to categorize
  --extension EXT   File extension to filter (e.g. .m4a)
  --help           Show this help message and exit
""")
        return

    trim_files_whitespaces(path, extension)
    files = read_files_from_path(path, extension)

    missing_files: set[str] = set()

    threshold = 50

    last_run_time = float("inf")

    while files:
        current_files = [file for file in files if file.name not in missing_files]
        if not current_files:
            break

        start_time = time.time()

        batch = current_files[:threshold]
        categorized = await categorize(batch)
        files = files[threshold:]

        for entry in categorized:
            category = entry.category
            filename = "".join([entry.name, extension])

            new_folder = os.path.join(path, category)
            os.makedirs(new_folder, exist_ok=True)

            old_path = os.path.join(path, filename)
            new_path = os.path.join(new_folder, filename)

            try:
                if os.path.exists(old_path):
                    shutil.move(old_path, new_path)
                    print(
                        f'Book \033[94m"{entry.name}"\033[0m moved to \033[92m"{entry.category}"\033[0m'
                    )
                else:
                    print(f"Warning: File not found - \033[91m{entry.name}\033[0m")
                    missing_files.add(entry.name)
            except Exception as e:
                print(
                    f"Error moving file \033[93m{filename}\033[0m: \033[91m{str(e)}\033[0m"
                )

        run_time = time.time() - start_time

        print(f"\nBatch completed in \033[92m{run_time:.2f} seconds\033[0m")

        if run_time < last_run_time:
            threshold += 10
            print(
                f"\033[94mRun was faster!\033[0m Increasing threshold to \033[93m{threshold}\033[0m\n"
            )

        last_run_time = run_time


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
