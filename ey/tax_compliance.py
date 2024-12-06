from crewai import Crew, Task, Agent, Process
from crewai_tools import WebsiteSearchTool, FileReadTool, FileWriterTool
import os
from crewai_tools import BaseTool

os.environ["OPENAI_API_KEY"] = ""

fileread = FileReadTool()
filewrite = FileWriterTool()
web_search = WebsiteSearchTool()
import os

class AskHuman(BaseTool):
    name : str = "Ask Human"
    description : str = "This tool is used to ask the expert human for assistance whenever necessary. This tool will facilitate the communication between agent and human. Please ask your query by passing it as an argument to the run method"

    def _run(self, query:str):
        return input(f"Please answer this question : {query}: \n" )






compliance_checker = Agent(
    role="Tax compliance verifier",
    goal="Given the tax audit of the client, read and make sure they comply with the rules by asking the rule gathering agent",
    backstory="An experienced tax compliance specialist who knows how to look for potential loopholes or mistakes in tax audit reports, especially in compliance.",
    tools=[fileread, AskHuman(), web_search],
    llm="gpt-4o",
    verbose=True
)

compliance_report_writer = Agent(
    role="Compliance report writing agent",
    goal="Write a detailed tax compliance report based on the analysis of compliance checking officer",
    backstory="An experienced financial writer specializing in writing tax compliance verification reports",
    tools=[fileread],
    llm='gpt-4o',
    verbose=True
)

manager = Agent(
    role="Team Lead",
    goal="Efficiently delegate the tasks of tax compliance verification and get the final report",
    backstory="An experienced team lead with many years in the compliance industry",
    allow_delegation=True,
    verbose=True,
    llm='gpt-4o'
)


task = Task(description="Generate the tax compliance verification report based on the provided tax audit at 'taxaudit.txt' and compliance rules at 'compliancerules.txt'. Please refer to the internet or ask the expert human for input whenever necessary",
            expected_output="""A .txt file that has the detailed analysis of the tax compliance verification in the following format:
            Analysis:
            Discrepancies found:
            potential new compliance rules from web:
            final remarks:
            """,
            output_file="compliance_report.txt"
            )


crew = Crew(
    agents=[compliance_checker, compliance_report_writer],
    tasks=[task],
    manager_agent=manager,
    process=Process.hierarchical,
    verbose=True
)

crew.kickoff()
