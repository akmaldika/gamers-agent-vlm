import os

def merge_txt_files(folder_path, output_path=None):
    result = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Format as requested
            formatted = f"{filename}\n---\n{content}\n---\n"
            result.append(formatted)

    # Combine all formatted parts
    final_result = "\n".join(result)

    # Save to output file if output_path is provided
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_result)

    return final_result


# Example usage
folder = "data/map_dataset/01_navigation"
output = "merged_output.txt"  # or None if you don't want to save
print(merge_txt_files(folder, output))
