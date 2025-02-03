import json
import os
import shutil
import time

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class Entry(BaseModel):
    """Entry representing something, with a name, a category and a type."""

    name: str = Field(description="The name of the entry")
    category: str = Field(description="The category of the entry")
    type: str = Field(description="The type of the entry")


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
        categories = json.load(file)

    return set(categories)


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

    categories = load_categories("app/categories.json")
    categories_str = ", ".join(categories)

    parser = PydanticOutputParser(
        pydantic_object=Entry,
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retry_parser = RetryOutputParser.from_llm(parser=parser, llm=llm)

    prompt = PromptTemplate(
        template="""
        You must ONLY select from these exact {type} categories: {categories_str}
        For the {type} titled "{entry}", return ONLY ONE category from the list above that best matches.
        DO NOT create new categories or modify existing ones.
        If no category fits perfectly, choose the closest match from the provided list.
        {format_instructions}
        """,
        input_variables=["categories_str", "entry", "type"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt_and_model = prompt | llm

    output = prompt_and_model.batch(
        inputs=[
            {"categories_str": categories_str, "entry": entry, "type": "book"}
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


async def main() -> None:
    """Main function to load and display categories."""

    path = "C:\\Users\\Eduardo\\Proton Drive\\My files\\Education\\Books\\Audiobooks\\Blinkist\\Blinkist August 2023 SiteRip Collection - BASiQ"
    extension = ".m4a"

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
