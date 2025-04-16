import re

def main():
    # Specify the path to your file here or input it at runtime.
    file_path = input("Enter path to the file: ").strip()
    
    # Regular expression to extract the times from lines that start with "Speed:"
    # It looks for numbers (which may include decimals) before "ms preprocess,", "ms inference," and "ms postprocess"
    pattern = r"Speed:\s*([\d.]+)ms preprocess,\s*([\d.]+)ms inference,\s*([\d.]+)ms postprocess"
    
    preprocess_total = 0.0
    inference_total = 0.0
    postprocess_total = 0.0
    count = 0
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Only process lines that contain the speed information
                match = re.search(pattern, line)
                if match:
                    # Convert captured groups (times) to floats
                    preprocess_time = float(match.group(1))
                    inference_time = float(match.group(2))
                    postprocess_time = float(match.group(3))
                    
                    # Accumulate totals and count the number of valid lines
                    preprocess_total += preprocess_time
                    inference_total += inference_time
                    postprocess_total += postprocess_time
                    count += 1
        
        if count > 0:
            avg_preprocess = preprocess_total / count
            avg_inference = inference_total / count
            avg_postprocess = postprocess_total / count
            # Total average time per line is the sum of the three averages.
            avg_total = avg_preprocess + avg_inference + avg_postprocess

            print("\nAverages based on {} valid entries:".format(count))
            print("Average preprocess time: {:.2f}ms".format(avg_preprocess))
            print("Average inference time: {:.2f}ms".format(avg_inference))
            print("Average postprocess time: {:.2f}ms".format(avg_postprocess))
            print("Average total time per line: {:.2f}ms".format(avg_total))
        else:
            print("No valid speed lines were found in the file.")
    
    except FileNotFoundError:
        print("Error: The file was not found. Please check the path and try again.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
