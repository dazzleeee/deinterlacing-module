import os

output_file = "project_code.txt"

# 你想要排除的文件夹名称列表
exclude_dirs = ["data_process", ".git", "__pycache__", "work_dirs"]

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk("."):
        # 核心修改：在原地修改 dirs 列表，剔除掉不需要进入的文件夹
        # 这样 os.walk 就根本不会去扫描这些文件夹内部的内容
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            # 顺便排除掉可能的隐藏文件或者这个合并脚本本身
            if file.endswith((".py", ".yaml")) and file != "addTogether.py":
                filepath = os.path.join(root, file)
                
                outfile.write(f"\n{'='*50}\n")
                outfile.write(f"FILE: {filepath}\n")
                outfile.write(f"{'='*50}\n")
                
                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"读取文件失败: {e}\n")

print(f"代码合并完成，已排除 {exclude_dirs} 目录，请把 {output_file} 传给 Gemini！")