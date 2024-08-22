import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq

# Set environment variables
os.environ["SERPER_API_KEY"] = "xxxx"

# Define the Crew AI agents and tasks
groq_api_key = 'xxx'
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0)

search_tool = SerperDevTool()

# Define the Financial Analyst Agent for PSU Stocks
analyst_psu = Agent(
    llm=llm,
    role="Senior Financial Analyst",
    goal="Identify top PSU stocks less then ₹200 on NSE for investment.",
    backstory="You are a seasoned financial analyst with a focus on identifying promising PSU stocks in the Indian stock market.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True,
)

# Define Task 1 for the PSU Analyst
task1_psu = Task(
    description="Search the internet and find 5 PSU stocks on the NSE. For each stock, provide the current price, P/E ratio, earnings per share (EPS), and any relevant financial indicators or news that would be useful for investment decisions.",
    expected_output="""A detailed report of each of the PSU stocks. The results should be formatted as shown below:

    Stock 1: XYZ Ltd.
    Current Price: ₹180
    P/E Ratio: 14.2
    EPS: ₹8.50
    Background Information: This PSU has shown consistent performance in the sector with recent improvements in operational efficiency. The following list highlights some of the top contenders for investment opportunities among PSU stocks.""",
    agent=analyst_psu,
    output_file="task1_psu_output.txt",
)

# Define the Writer Agent for PSU Stocks
writer_psu = Agent(
    llm=llm,
    role="Senior Financial Analyst",
    goal="Summarise PSU stock information into a report for investors.",
    backstory="You are a financial analyst, your goal is to compile PSU stock analytics into a report for potential investors.",
    allow_delegation=False,
    verbose=True,
)

# Define Task 2 for the PSU Writer
task2_psu = Task(
    description="Summarise the PSU stock information into a bullet point list.",
    expected_output="A summarised dot point list of each PSU stock, prices, P/E ratio, EPS, and important features of that stock.",
    agent=writer_psu,
    output_file="task2_psu_output.txt",
)

# Initialize Crew with the PSU Agents and Tasks
crew = Crew(agents=[analyst_psu, writer_psu], tasks=[task1_psu, task2_psu], verbose=True)

# Kick off the tasks and print the output
task_output = crew.kickoff()
print(task_output)
