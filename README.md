# AI Folderizer

AI-powered Python tool that automatically categorizes and organizes files into folders using OpenAI's GPT models.

## Features

- Automatically categorizes files using GPT-4 and semantic similarity
- Moves files into category-based folders
- Supports batch processing with dynamic batch size optimization
- Handles file name whitespace trimming
- Built-in retry mechanism for API calls
- Example-based learning with semantic similarity matching

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/eduardohilariodev/ai-folderizer.git
   ```

2. Navigate into the project directory:

   ```bash
   cd ai-folderizer
   ```

3. Install dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables by copying the example file:

   ```bash
   cp .env.example .env
   ```

5. Update the `.env` file with your OpenAI API key and LangSmith if you want to.

## Usage

To use the AI Folderizer, follow these steps:

1. **Prepare Your Categories**:
   - Edit the `app/categories.json` file to include the categories you want to use for organizing your files. The file should be a JSON array of category names. For example:

     ```json
     [
       "Category1",
       "Category2",
       "Category3"
     ]
     ```

2. **Run the Main Script**:
   - Execute the script with the required parameters:

     ```bash
     python app/main.py --path /path/to/files --extension .ext
     ```

   Parameters:
   - `--path`: Directory path containing files to categorize
   - `--extension`: File extension to filter (e.g. .m4a)
   - `--help`: Show help message and exit

3. **Check the Output**:
   - After running the script, check the specified directory for newly created folders corresponding to the categories. The files will be moved into their respective category folders.

4. **Environment Variables**:
   - Ensure that your environment variables are set correctly in the `.env` file, especially the `OPENAI_API_KEY`, to allow the script to access the OpenAI API for categorization.

## Example

To categorize audio files in the "Audiobooks" directory:
