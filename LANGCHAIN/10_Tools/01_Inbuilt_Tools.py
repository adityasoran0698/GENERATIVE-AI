from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

# search_tool = DuckDuckGoSearchRun()

# result1 = search_tool.invoke("IPL 2026")
# print(result1)

shell_tool = ShellTool()

result2 = shell_tool.invoke("whoami")
print(result2)
