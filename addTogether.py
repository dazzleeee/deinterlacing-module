import os

output_file = "project_code.txt"

# Manual exclusions for known irrelevant folders
exclude_dirs = ["__pycache__", "work_dirs", "data_process"]

# File extensions to include
include_exts = (".py", ".sh", ".yaml")

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk("."):
        # Skip hidden folders and manual exclusions
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in exclude_dirs]

        for file in files:
            # Include only desired file types, skip this merge script itself
            if file.endswith(include_exts) and file != "addTogether.py":
                filepath = os.path.join(root, file)
                
                outfile.write(f"\n{'='*50}\n")
                outfile.write(f"FILE: {filepath}\n")
                outfile.write(f"{'='*50}\n")
                
                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"读取文件失败: {e}\n")

print(f"代码合并完成，已排除隐藏文件夹和 {exclude_dirs} 目录，请把 {output_file} 传给 Gemini！")