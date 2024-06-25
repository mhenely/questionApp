# OpenAI Document Loader Question App

This application allows the user to upload files and ask the AI questions about the content 

Currently, three official document formats are supported:

- .pdf
- .docx
- .txt

## How to use

- Clone the repo and run npm install in the root directory
- install dependencies and then use the command streamlit \frontend.py to run the application

Insert your OpenAI Api Key and then either drag and drop your file on click browse to select your file

- Adjust your chunk size and k value 
- Click Add Data once you are done and wait for the file to read, chunk, and embed the file for use
- Once embedding is completed, ask the AI any series of questions about your file that you wish!

![Alt text](Assets/questionAppScreenshot.png 'Questions about the Hobbit')


## In development

- Additional supported file formats
- Access to external sources (ex. wikipedia)
