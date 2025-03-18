import os

# absolute_path = os.path.abspath("admin/test1.mp3")
# print(f"Resolved Absolute Path: {absolute_path}")
# print("hello world")

import os

file_path = "C:\\Users\\bwgya\\Documents\\GitHub\\AudioAuditRAG_2025P1\\admin\\test1.mp3"

if not os.path.exists(file_path):
    print(f"Error: File does not exist - {file_path}")
else:
    print(f"File exists and is accessible: {file_path}")