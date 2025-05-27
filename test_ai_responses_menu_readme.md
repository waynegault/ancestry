# AI Response Testing Script with Menu Interface

This script is designed to help test and refine AI prompts for genealogical responses. It creates a test database with fictitious people and messages, processes these messages using the existing codebase functions, and allows you to evaluate and improve the AI-generated responses.

## Overview

The script provides a menu-based interface for the following functions:

1. **Create Test Database and Generate Messages**: Creates a new SQLite database with 100 fictitious people and varied message types.
2. **Process Messages and Generate AI Responses**: Uses the existing AI functionality to process messages and generate responses.
3. **Evaluate AI Responses**: Allows you to review each response and mark it as acceptable (1) or unacceptable (0).
4. **Analyze Feedback and Suggest Improvements**: Analyzes patterns in acceptable vs. unacceptable responses and suggests improvements.
5. **Update AI Prompts Based on Feedback**: Generates improved AI prompts based on the feedback and saves them to files.

## Requirements

- Python 3.6+
- All dependencies from the main codebase
- SQLite

## Usage

Simply run the script and follow the menu prompts:

```bash
python test_ai_responses_menu.py
```

### Menu Options

The script will display a menu with the following options:

```
========================================================================
                    AI RESPONSE TESTING MENU
========================================================================

1. Create Test Database and Generate Messages
2. Process Messages and Generate AI Responses
3. Evaluate AI Responses
4. Analyze Feedback and Suggest Improvements
5. Update AI Prompts Based on Feedback
6. Exit

========================================================================
```

Enter the number corresponding to the action you want to perform.

### Typical Workflow

1. Start by selecting option 1 to create the test database and generate messages.
2. Select option 2 to process the messages and generate AI responses.
3. Select option 3 to evaluate the responses, marking each as acceptable (1) or unacceptable (0).
4. Select option 4 to analyze the feedback and get suggestions for improvements.
5. Select option 5 to generate improved prompts based on the feedback.
6. Repeat steps 2-5 as needed to refine the prompts.

## Message Types

The script generates various types of fictitious messages:

1. **Not Interested** (10%): Messages indicating the person is not interested in genealogy research.
2. **Interested but Brief** (10%): Short messages expressing interest but without specific details.
3. **Detailed In-Tree** (10%): Detailed messages about ancestors who are in the user's family tree.
4. **Detailed Not-In-Tree** (10%): Detailed messages about ancestors who are not in the user's family tree.
5. **Partial Details In-Tree** (10%): Messages with partial information about ancestors in the tree.
6. **Miscellaneous** (50%): Various other genealogy-related messages.

## Output Files

- `data/test_ai_responses.db`: SQLite database containing test data and responses
- `improved_prompts/improved_extraction_prompt.txt`: Improved prompt for extracting information from messages
- `improved_prompts/improved_response_prompt.txt`: Improved prompt for generating responses

## Notes

- The test database is stored in `data/test_ai_responses.db` and will be reused if it exists.
- The script uses the same AI providers and configuration as the main codebase.
- The evaluation process is interactive and requires manual input.
- The analysis and prompt improvement are automated based on the evaluation feedback.
- You can exit the script at any time by selecting option 6 from the menu.

## Troubleshooting

If you encounter any issues:

1. Check the log file for error messages.
2. Ensure all dependencies are installed.
3. Verify that the database file is accessible and not corrupted.
4. Make sure the AI providers are properly configured in the main codebase.

If the script fails to connect to an existing database, try deleting the database file and creating a new one (option 1).
