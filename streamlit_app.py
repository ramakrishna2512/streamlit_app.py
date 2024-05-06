import streamlit as st
from transformers import pipeline

def main():
    st.title("Question Answering App")

    # Load the QA pipeline
    model_name = 'deepset/roberta-base-squad2'
    qa = pipeline('question-answering', model=model_name, tokenizer=model_name)

    # Define the default example questions and contexts
    qa_input = [{'question': 'What is the purpose of using an SQL SELECT statement?',
                 'context': 'The SQL SELECT statement is used to retrieve data from a database. It allows you to specify which columns to retrieve and which table or tables to retrieve them from, as well as any conditions that must be met for the data to be included in the result set.'},
                {'question': 'How can you filter the results of an SQL SELECT statement?',
                 'context': 'You can filter the results of an SQL SELECT statement using the WHERE clause. The WHERE clause allows you to specify conditions that must be met for a row to be included in the result set. For example, you could use the WHERE clause to only retrieve rows where a certain column is equal to a specific value.'}]

    # Display the default example questions and contexts
    for i, item in enumerate(qa_input):
        st.markdown(f"### Example {i+1}:")
        st.markdown(f"**Question:** {item['question']}")
        st.markdown(f"**Context:** {item['context']}")
    
    st.markdown("---")

    # Get user input for question and context
    user_question = st.text_input("Enter your question:")
    user_context = st.text_area("Enter the context:")

    # Process the user input and display the answer
    if st.button("Get Answer"):
        output = qa(question=user_question, context=user_context)
        st.markdown(f"**Answer:** {output['answer']}")
        st.markdown(f"**Score:** {output['score']}")

if __name__ == "__main__":
    main()
