import os
from openai import OpenAI
import constants

client = OpenAI(
        api_key=constants.APIKEY,
    )

def text_to_json(file):
    # Create an empty list to store JSON objects
    json_objects = []

    completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": constants.INPUT_FORMATTER_PROMPT},
                {"role": "user", "content": constants.USER_FORMATTER_PROMPT+"\n"+file}
            ],
            model="gpt-3.5-turbo",
            max_tokens=800,  # Adjust as needed
            temperature=0.1
    )
    # Get the response
    response = completion.choices[0].message.content.strip().replace("\n", "")
    json_objects.append(response)

    return json_objects

def gen_feedback(inputs,output_file):
    # Open the output file in append mode
    with open(output_file, "a") as file:
        # Loop through the inputs
        for input_text in inputs:
            # Create completion request
            completion = client.chat.completions.create(
            messages=[
                    {"role": "system", "content": constants.FEEDBACK_GIVER_PROMT},
                    {"role": "user", "content": constants.USER_PROMPT+"\n"+input_text}
            ],
            model="gpt-3.5-turbo",
            max_tokens=1500,  # Adjust as needed
            temperature=0.6
    )

            # Get the response 
            response = completion.choices[0].message.content

            # Write the response to the output file
            file.write(f"Input: {input_text}\n")
            file.write(f"Response: {response}\n\n")
            file.write("_________________NEXT_________________\n\n")


            # Print response to console
            print(f"Input\n: {input_text}")
            print(f"Response\n: {response}\n")
            print("_________________NEXT_____________________\n\n")

    print(f"Output written to {output_file}")

def main():

    # Directory containing files
    directory = './inputs'

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # Process only .txt files
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                # Construct object from text
                inputs = text_to_json(text)
                gen_feedback(inputs,'output-'+filename)


if __name__ == "__main__":
    main()