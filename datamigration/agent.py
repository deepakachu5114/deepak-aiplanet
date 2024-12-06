# Standard library imports
import asyncio
import json
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Literal

# Third-party imports
import pandas as pd
from sqlalchemy import create_engine, text
from IPython.display import Image, display
from rich.console import Console

# LangChain imports
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.runnables.graph import MermaidDrawMethod

import re

# Type definitions
class DBAgentState(TypedDict):
    messages: List[Any]
    mysql_schema: Dict[str, str]
    postgresql_schema: Any
    errors: List[Dict[str, str | Any]]
    feedback: List[str]
    review_status: Literal["pending", "approved", "needs_revision"]
    table_data: Dict[Any, Any]
    row_counts: Dict[Any, Any]
    dependency_attempts : int
    table_order : Any


# Global setup
console = Console(force_terminal=True)  # Force terminal output


def log_message(message: str, style: str = "default"):
    """Utility function to ensure console messages are displayed"""
    console.print(message, style=style, markup=True)
    console.print("")  # Add blank line for better readability


def db_execute_query(db, query: str) -> str:
    """Execute a SQL query against the database."""
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result

def schema_extraction_node(state: DBAgentState) -> DBAgentState:
    """Node that extracts schema information"""
    log_message("[bold blue]Starting Schema Extraction...[/bold blue]")

    try:
        tables = get_table_names.run("").split(", ")
        log_message(f"Found {len(tables)} tables to process")

        schema_dict = {}
        for table in tables:
            log_message(f"Getting schema for [cyan]{table}[/cyan]")
            schema_dict[table] = get_schema.run(table)
            log_message(f"✓ Schema extracted for [green]{table}[/green]")

        return {
            "messages": state.get("messages", []) + [
                HumanMessage(content="Schema extraction complete")
            ],
            "mysql_schema": schema_dict,
            "postgresql_schema": "",
            "errors": state.get("errors", []),
            "feedback": state.get("feedback", []),
            "review_status": "pending",
            "table_data": state.get("table_data", {}),
            "row_counts": state.get("row_counts", {}),
            "dependency_attempts" : state.get("dependency_attempts", 0),
            "table_order" : []
        }
    except Exception as e:
        log_message(f"[red]Error during schema extraction: {str(e)}[/red]")
        raise


def create_postgres_converter_chain(llm):
    """Creates a chain for converting MySQL to PostgreSQL schema"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a database expert specializing in converting MySQL schemas to PostgreSQL.
        Convert the provided MySQL schema to PostgreSQL following these rules:
        1. Handle all MySQL data types appropriately
        2. Convert auto_increment to SERIAL or IDENTITY
        3. Replace backticks with double quotes for identifiers
        4. Remove MySQL-specific storage parameters (ENGINE, CHARSET, etc.)
        5. Combine all table definitions into a single, valid PostgreSQL script
        6. Include appropriate foreign key relationships
        7. Maintain the original table relationships and constraints

        Return ONLY the converted PostgreSQL schema as a single SQL script, with no explanations or markdown."""),
        ("human", """Please convert these MySQL table schemas to PostgreSQL:
        {mysql_schemas}
        Remember to combine all table definitions into a single script and maintain their relationships.""")
    ])
    return prompt | llm


def postgres_converter_node(state: DBAgentState, conversion_chain) -> DBAgentState:
    """Node that uses LLM to convert MySQL schema to PostgreSQL"""
    log_message("[bold blue]Starting PostgreSQL Schema Conversion...[/bold blue]")

    try:
        mysql_schemas = "\n\n".join(
            f"Table: {table}\n{schema}"
            for table, schema in state["mysql_schema"].items()
        )

        log_message("Sending schemas to LLM for conversion...")
        result = conversion_chain.invoke({"mysql_schemas": mysql_schemas})
        log_message("[green]✓ Schema conversion completed[/green]")

        return {
            "messages": state.get("messages", []) + [
                HumanMessage(content="Schema extraction complete")
            ],
            "mysql_schema": state.get("mysql_schema", {}),
            "postgresql_schema": result.content[6:-4],
            "errors": state.get("errors", []),
            "feedback": state.get("feedback", []),
            "review_status": "pending",
            "table_data": state.get("table_data", {}),
            "row_counts": state.get("row_counts", {}),
            "dependency_attempts" : state.get("dependency_attempts", 0),
            "table_order" : []
        }
    except Exception as e:
        log_message(f"[red]Error during schema conversion: {str(e)}[/red]")
        raise


def human_input():
    """Node that takes in the information about the source and destination databases"""
    log_message("\n[bold purple]Database Migration Setup[/bold purple]")

    # Get MySQL connection string
    mysql_conn_string = input("\nEnter the MySQL connection string: ")
    mysql_version = input("\nEnter the MySQL version: ")
    mydb = (input("\nEnter the MySQL database name: "))
    # Get PostgreSQL connection string
    postgresql_conn_string = input("\nEnter the PostgreSQL connection string: ")
    postgres_version = input("\nEnter the PostgreSQL version: ")
    posdb = (input("\nEnter the PostgreSQL database name: "))



    return mysql_conn_string, postgresql_conn_string, mydb, posdb

def human_review_node(state: DBAgentState) -> DBAgentState:
    """Node that handles human review of the schema conversion"""
    log_message("\n[bold purple]Schema Review Required[/bold purple]")

    # Display MySQL Schema
    log_message("\n[bold blue]MySQL Schema:[/bold blue]")
    for table, schema in state["mysql_schema"].items():
        log_message(f"\n-- Table: {table}")
        log_message(schema)

    # Display PostgreSQL Schema
    log_message("\n[bold green]PostgreSQL Schema:[/bold green]")
    log_message(state["postgresql_schema"])

    # Get feedback
    while True:
        response = input("\nIs the PostgreSQL schema correct? (yes/no): ").lower()
        if response in ['yes', 'y']:
            feedback = ""
            approved = True
            break
        elif response in ['no', 'n']:
            feedback = input("\nPlease provide feedback: \n")
            approved = False
            break
        else:
            log_message("[yellow]Please answer 'yes' or 'no'[/yellow]")

    new_state = {
        "messages": state.get("messages", []),
        "mysql_schema": state["mysql_schema"],
        "postgresql_schema": state["postgresql_schema"],
        "errors": state.get("errors", []),
        "feedback": state.get("feedback", []),
        "review_status": "approved" if approved else "needs_revision",
        "table_data": state.get("table_data", {}),
        "row_counts": state.get("row_counts", {}),
        "dependency_attempts": state.get("dependency_attempts", 0),
        "table_order": []
    }

    if approved:
        log_message("[green]✓ Schema approved[/green]")
        new_state["messages"].append(
            HumanMessage(content="Schema conversion approved by user")
        )
    else:
        log_message("[yellow]Schema needs revision[/yellow]")
        new_state["feedback"].append(feedback)
        new_state["messages"].append(
            HumanMessage(content=f"Schema conversion needs revision: {feedback}")
        )

    return new_state


def create_tables_node(state: DBAgentState, postgresql_conn_string: str) -> DBAgentState:
    """Node that creates PostgreSQL tables"""
    log_message("[bold blue]Creating PostgreSQL Tables...[/bold blue]")

    new_state = {
        "messages": state.get("messages", []),
        "mysql_schema": state["mysql_schema"],
        "postgresql_schema": state["postgresql_schema"],
        "errors": state.get("errors", []),
        "feedback": state.get("feedback", []),
        "review_status": state["review_status"],
        "table_data": state.get("table_data", {}),
        "row_counts": state.get("row_counts", {}),
        "dependency_attempts": state.get("dependency_attempts", 0),
        "table_order": []
    }

    try:
        pg_db = SQLDatabase.from_uri(postgresql_conn_string)
        schema_statements = [
            stmt.strip()
            for stmt in state["postgresql_schema"].split(';')
            if stmt.strip()
        ]

        for statement in schema_statements:
            if statement.lower().strip().startswith('create table'):
                log_message(f"\nExecuting: [dim]{statement[:100]}...[/dim]")
                db_execute_query(pg_db, statement)
        log_message("[bold green]All PostgreSQL tables created successfully![/bold green]")
        new_state["messages"].append(
            HumanMessage(content="PostgreSQL tables created successfully")
        )

    except Exception as e:
        log_message(f"[red]Error creating tables: {str(e)}[/red]")
        new_state["errors"].append({
            "error": str(e),
            "step": "table_creation"
        })
        raise

    return new_state


def extract_mysql_data_node(state: DBAgentState, mysql_conn_string: str) -> DBAgentState:
    """Node that extracts data from MySQL tables"""
    log_message("[bold blue]Starting MySQL Data Extraction...[/bold blue]")

    new_state = {
        "messages": state.get("messages", []),
        "mysql_schema": state["mysql_schema"],
        "postgresql_schema": state["postgresql_schema"],
        "errors": state.get("errors", []),
        "feedback": state.get("feedback", []),
        "review_status": state["review_status"],
        "table_data": state.get("table_data", {}),
        "row_counts": state.get("row_counts", {}),
        "dependency_attempts": state.get("dependency_attempts", 0),
        "table_order": []
    }

    try:
        mysql_engine = create_engine(mysql_conn_string)
        tables = list(state["mysql_schema"].keys())

        for table in tables:
            log_message(f"\nProcessing table: [cyan]{table}[/cyan]")

            try:
                # Get total count
                count_query = f"SELECT COUNT(*) FROM `{table}`"
                total_rows = pd.read_sql(count_query, mysql_engine).iloc[0, 0]
                new_state["row_counts"][table] = total_rows
                log_message(f"Total rows to extract: {total_rows}")

                # Read in chunks
                chunks = []
                chunk_count = 0
                for chunk_df in pd.read_sql_table(
                        table,
                        mysql_engine,
                        chunksize=10000
                ):
                    chunks.append(chunk_df)
                    chunk_count += 1
                    log_message(f"Extracted chunk {chunk_count} ({len(chunk_df)} rows)", style="dim")

                new_state["table_data"][table] = chunks
                log_message(f"[green]✓ Completed {table}: {total_rows} rows in {chunk_count} chunks[/green]")

            except Exception as e:
                log_message(f"[red]✗ Failed {table}: {str(e)}[/red]")
                new_state["errors"].append({
                    "table": table,
                    "error": str(e),
                    "step": "data_extraction"
                })

        log_message("[bold green]MySQL Data Extraction Complete![/bold green]")

    except Exception as e:
        log_message(f"[red]Error during MySQL extraction: {str(e)}[/red]")
        new_state["errors"].append({
            "error": str(e),
            "step": "mysql_extraction"
        })
        raise

    return new_state


def parse_postgresql_schema(schema_string: str) -> Dict:
    """
    Parse PostgreSQL schema string into a structured dictionary
    """
    # Remove newlines and standardize spacing
    schema = ' '.join(schema_string.split())

    # Split into individual CREATE TABLE statements
    table_statements = re.findall(r'CREATE TABLE "([^"]+)" \((.*?)\);', schema)

    schema_dict = {}

    for table_name, table_content in table_statements:
        # Split content into individual column/constraint definitions
        definitions = re.findall(r'(?:[^,]+(?:\([^)]*\))?)+(?:,|$)', table_content)

        table_info = {
            'columns': {},
            'primary_key': None,
            'foreign_keys': [],
            'checks': []
        }

        for def_str in definitions:
            def_str = def_str.strip()

            # Handle PRIMARY KEY
            if def_str.startswith('PRIMARY KEY'):
                pk_cols = re.findall(r'"([^"]+)"', def_str)
                table_info['primary_key'] = pk_cols

            # Handle FOREIGN KEY
            elif def_str.startswith('FOREIGN KEY'):
                fk_match = re.search(r'FOREIGN KEY \("([^"]+)"\) REFERENCES "([^"]+)" \("([^"]+)"\)', def_str)
                if fk_match:
                    column, ref_table, ref_column = fk_match.groups()
                    table_info['foreign_keys'].append({
                        'constraint_name': f'fk_{table_name}_{column}_{ref_table}',
                        'column': column,
                        'referenced_table': ref_table,
                        'referenced_column': ref_column
                    })

            # Handle CHECK constraints
            elif 'CHECK' in def_str:
                check_match = re.search(r'CHECK \((.*?)\)', def_str)
                if check_match:
                    table_info['checks'].append(check_match.group(1))

            # Handle regular columns
            else:
                col_match = re.match(r'"([^"]+)" ([^"]+?)(?:\s+NOT NULL)?(?:\s*,)?$', def_str)
                if col_match:
                    col_name, col_type = col_match.groups()
                    table_info['columns'][col_name] = {
                        'type': col_type.strip(),
                        'nullable': 'NOT NULL' not in def_str
                    }

        schema_dict[table_name] = table_info

    return schema_dict


def create_dependency_resolver_chain(llm):
    """Creates a chain for determining table loading order"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a database expert specializing in determining the correct order to load tables based on their foreign key dependencies.

        Given a PostgreSQL schema and optionally any previous errors, determine the optimal order to load the tables.

        Rules:
        1. Tables referenced by foreign keys must be loaded before tables that reference them
        2. If you see any errors from previous attempts, adjust the order to resolve those specific issues
        3. Consider both direct and indirect dependencies
        4. Parent tables (those being referenced) should come before child tables (those with foreign keys)

        Return ONLY a Python list of table names in the correct loading order, with no explanations or additional text."""),
        ("human", """PostgreSQL Schema:
        {schema}

        Previous Errors (if any):
        {errors}

        Return the correct table loading order as a Python list.""")
    ])
    return prompt | llm


def solve_dependencies_node(state: DBAgentState) -> DBAgentState:
    """Node that uses LLM to determine correct table loading order"""
    log_message("[bold blue]Determining Table Loading Order...[/bold blue]")

    try:
        # Create resolver chain
        resolver_chain = create_dependency_resolver_chain(llm)

        # Format any previous errors
        previous_errors = "\n".join([
            f"Error loading {e.get('table', 'unknown table')}: {e.get('error', '')}"
            for e in state.get("errors", [])
            if e.get("step") == "data_loading"
        ])

        # Get loading order from LLM
        result = resolver_chain.invoke({
            "schema": state["postgresql_schema"],
            "errors": previous_errors if previous_errors else "No previous errors"
        })

        # Extract the list from the response
        try:
            # Clean and evaluate the response to get the Python list
            table_order = eval(result.content.strip()[10:-4])
            if not isinstance(table_order, list):
                raise ValueError("Response is not a list")
        except Exception as e:
            log_message(f"[red]Error parsing LLM response: {str(e)}[/red]")
            raise

        log_message(f"Determined loading order: [green]{', '.join(table_order)}[/green]")

        # Update state with the loading order
        new_state = {"messages": state.get("messages", []), "mysql_schema": state["mysql_schema"],
                     "postgresql_schema": state["postgresql_schema"], "errors": state.get("errors", []),
                     "feedback": state.get("feedback", []), "review_status": state["review_status"],
                     "table_data": state.get("table_data", {}), "row_counts": state.get("row_counts", {}),
                     "dependency_attempts": state.get("dependency_attempts", 0) + 1, "table_order": table_order}

        return new_state

    except Exception as e:
        log_message(f"[red]Error determining table order: {str(e)}[/red]")
        raise


def load_postgresql_data_node(state: DBAgentState, postgresql_conn_string: str) -> dict:
    """Node that loads data into PostgreSQL tables using LLM-determined loading order"""
    log_message("[bold blue]Starting PostgreSQL Data Loading...[/bold blue]")

    new_state = {
        "messages": state.get("messages", []),
        "mysql_schema": state["mysql_schema"],
        "postgresql_schema": state["postgresql_schema"],
        "errors": [],  # Reset errors for this attempt
        "feedback": state.get("feedback", []),
        "review_status": state["review_status"],
        "table_data": state.get("table_data", {}),
        "row_counts": state.get("row_counts", {}),
        "dependency_attempts": state.get("dependency_attempts", 0),
        "table_loading_order": state.get("table_order", [])
    }

    try:
        pg_engine = create_engine(postgresql_conn_string)

        # Use the LLM-determined loading order
        tables_to_load = [t for t in state["table_order"] if t in state["table_data"]]

        log_message(f"Loading tables in order: {', '.join(tables_to_load)}")

        # Temporarily disable foreign key constraints
        with pg_engine.connect() as connection:
            connection.execute(text("SET session_replication_role = 'replica';"))
            connection.commit()

        # Load tables in the LLM-determined order
        for table in tables_to_load:
            chunks = state["table_data"][table]
            log_message(f"\nLoading table: [cyan]{table}[/cyan]")
            log_message(f"Total chunks to load: {len(chunks)}")

            try:
                # Drop table if exists
                with pg_engine.connect() as connection:
                    connection.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
                    connection.commit()

                # Load data chunks
                for i, chunk_df in enumerate(chunks):
                    log_message(f"Loading chunk {i + 1}/{len(chunks)} ({len(chunk_df)} rows)", style="dim")
                    chunk_df.to_sql(
                        table,
                        pg_engine,
                        if_exists='append' if i > 0 else 'replace',
                        index=False,
                        method='multi',
                        chunksize=10000
                    )

                # Verify row count
                pg_count = pd.read_sql(f'SELECT COUNT(*) FROM "{table}"', pg_engine).iloc[0, 0]
                if pg_count != state["row_counts"][table]:
                    error_msg = f"Row count mismatch. MySQL: {state['row_counts'][table]}, PostgreSQL: {pg_count}"
                    log_message(f"[red]✗ {error_msg}[/red]")
                    new_state["errors"].append({
                        "table": table,
                        "error": error_msg,
                        "step": "data_verification"
                    })
                else:
                    log_message(f"[green]✓ Completed {table}: {pg_count} rows loaded successfully[/green]")

            except Exception as e:
                error_msg = str(e)
                log_message(f"[red]✗ Failed {table}: {error_msg}[/red]")
                new_state["errors"].append({
                    "table": table,
                    "error": error_msg,
                    "step": "data_loading"
                })
                # Early return on error to trigger dependency resolution
                return new_state

        # Re-enable constraints and add foreign keys
        with pg_engine.connect() as connection:
            # Restore normal role settings
            connection.execute(text("SET session_replication_role = 'origin';"))
            connection.commit()

            # Add foreign key constraints
            schema_dict = parse_postgresql_schema(state["postgresql_schema"])
            for table_name, table_info in schema_dict.items():
                if table_name in tables_to_load and table_info['foreign_keys']:
                    for fk in table_info['foreign_keys']:
                        constraint_sql = f"""
                        ALTER TABLE "{table_name}"
                        ADD CONSTRAINT "{fk['constraint_name']}"
                        FOREIGN KEY ("{fk['column']}")
                        REFERENCES "{fk['referenced_table']}" ("{fk['referenced_column']}")
                        ON DELETE CASCADE
                        """

                        try:
                            connection.execute(text(constraint_sql))
                            connection.commit()
                            log_message(f"[green]✓ Added foreign key {fk['constraint_name']} to {table_name}[/green]")
                        except Exception as e:
                            log_message(
                                f"[yellow]Warning: Failed to add constraint {fk['constraint_name']}: {str(e)}[/yellow]")
                            new_state["errors"].append({
                                "table": table_name,
                                "error": str(e),
                                "step": "constraint_creation"
                            })

        log_message("[bold green]PostgreSQL Data Loading Complete![/bold green]")

    except Exception as e:
        error_msg = str(e)
        log_message(f"[red]Error during PostgreSQL loading: {error_msg}[/red]")
        new_state["errors"].append({
            "error": error_msg,
            "step": "postgresql_loading"
        })

    return new_state


def compare_llm_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a database expert specializing in data verification between MySQL and PostgreSQL databases.
           Given the results from qurying a mySQL db and a postgreSQL db both, compare them and classify into:
           
           a) match
           b) mismatch
           c) error
        
            If two results match and are largely consistent, then return match. If you spot any huge mismatches that are indicative of inconsistency between the databases, 
            then return mismatch. In case the queries have thrown errors, return error.

           Format your response as JSON with this structure:
           {{
                "result" : <appropriate classification tag>,
                "reason" : <reasoning behind assigning the tag>
           }}"""),
        ("human", """Compare the results for these 2 query results:
            
            query description:
            {query_desc}
            
           MySQL query result:
           {mysql_result}

           PostgreSQL query result:
           {postgresql_result}
           """)
    ])

    return prompt | llm

def compare_query_results(mysql_query: str, postgres_query: str, query_desc: str) -> str:
    """
    Execute queries on both databases and compare results.
    Returns comparison results or error message.
    """
    try:
        # Execute MySQL query
        mysql_engine = create_engine("mysql+pymysql://root:deepak@localhost/employees")
        mysql_result = pd.read_sql(mysql_query, mysql_engine)

        # Execute PostgreSQL query
        pg_engine = create_engine("postgresql://deepakachu:deepak@localhost:5432/employees")
        postgres_result = pd.read_sql(postgres_query, pg_engine)
        print(mysql_result)
        print(postgres_result)

        # Compare results
        comparator_chain = compare_llm_chain(llm)
        text = comparator_chain.invoke(
            {
            "query_desc" : query_desc,
            "mysql_result" : mysql_result,
            "postgresql_result" : postgres_result
            }
        )
        test_result = json.loads(text.content.strip()[7:-3])

        if test_result["result"] == "match":
            return (f"""
                     Query Results Match ✓
                     Query : {query_desc}
                     MySQL : {len(mysql_result)}
                     PostgreSQL : {len(postgres_result)}
                     Comments: {test_result["reason"]}        
                    """)

        elif test_result["result"] == "mismatch":
            return (f"""
                    Query Results Mismatch ✗
                    Query : {query_desc}
                    MySQL : {len(mysql_result)}
                    PostgreSQL : {len(postgres_result)}
                    Comments: {test_result["reason"]} 
                        """)
        else:
            return (
                f"""
                There was an error
                Query : {query_desc}
                Comments: {test_result["reason"]} 
                """
            )
    except Exception as e:
        return f"Error executing queries: {str(e)}"


def create_verification_chain(llm):
    """Creates a chain for generating verification queries"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a database expert specializing in data verification between MySQL and PostgreSQL databases.
        Given the schemas for both databases, generate pairs of equivalent queries that will help verify data consistency.

        Focus on:
        1. Row counts and basic integrity checks
        2. Data distribution checks (min, max, averages)
        3. Relationship verification (foreign key integrity)
        4. Data type consistency
        5. Null value consistency
        6. Unique constraint verification

        For each verification, provide:
        1. A descriptive name for the check
        2. The MySQL query
        3. The equivalent PostgreSQL query
        4. What the check verifies

        Format your response as JSON with this structure:
        {{
            "verifications": [
                {{
                    "name": "check name",
                    "description": "what this check verifies",
                    "mysql_query": "mysql query here",
                    "postgresql_query": "postgresql query here"
                }}
            ]
        }}"""),
        ("human", """Generate verification queries for these schemas:

        MySQL Schema:
        {mysql_schema}

        PostgreSQL Schema:
        {postgresql_schema}
        """)
    ])

    return prompt | llm


def verification_node(state: DBAgentState) -> dict:
    """Node that performs data verification between MySQL and PostgreSQL"""
    log_message("[bold blue]Starting Data Verification...[/bold blue]")

    new_state = {
        "messages": state.get("messages", []),
        "mysql_schema": state["mysql_schema"],
        "postgresql_schema": state["postgresql_schema"],
        "errors": state.get("errors", []),
        "feedback": state.get("feedback", []),
        "review_status": state["review_status"],
        "table_data": state.get("table_data", {}),
        "row_counts": state.get("row_counts", {}),
        "dependency_attempts": state.get("dependency_attempts", 0)

    }
    new_state.setdefault("verification_results", [])

    try:
        # Create verification chain
        verification_chain = create_verification_chain(llm)

        # Get verification queries
        mysql_schemas = "\n\n".join(
            f"Table: {table}\n{schema}"
            for table, schema in state["mysql_schema"].items()
        )

        result = verification_chain.invoke({
            "mysql_schema": mysql_schemas,
            "postgresql_schema": state["postgresql_schema"]
        })

        # Parse the JSON
        print(result.content.strip())
        verifications = json.loads(result.content.strip()[7:-3])["verifications"]

        log_message(f"\nRunning {len(verifications)} verification checks...")

        for verification in verifications:
            log_message(f"\n[bold]Running: {verification['name']}[/bold]")
            log_message(f"Description: {verification['description']}")

            try:
                result = compare_query_results(
                    verification["mysql_query"],
                    verification["postgresql_query"],
                    verification["description"]
                )

                verification_result = {
                    "name": verification["name"],
                    "result": result,
                    "status": "success" if "match" in result else "mismatch"
                }

                new_state["verification_results"].append(verification_result)

                # Display result
                if "Match" in result:
                    log_message(f"[green]{result}[/green]")
                else:
                    log_message(f"[red]{result}[/red]")

            except Exception as e:
                log_message(f"[red]Error in verification {verification['name']}: {str(e)}[/red]")
                new_state["verification_results"].append({
                    "name": verification["name"],
                    "description": verification["description"],
                    "error": str(e),
                    "status": "error"
                })

        # Summarize results
        successes = sum(1 for v in new_state["verification_results"] if v["status"] == "success")
        mismatches = sum(1 for v in new_state["verification_results"] if v["status"] == "mismatch")
        errors = sum(1 for v in new_state["verification_results"] if v["status"] == "error")

        log_message(f"""
                    [bold]Verification Summary:[/bold]
                    ✓ Successful matches: {successes}
                    ✗ Mismatches found: {mismatches}
                    ! Errors encountered: {errors}
                    """)

    except Exception as e:
        log_message(f"[red]Error during verification: {str(e)}[/red]")
        new_state["errors"].append({
            "error": str(e),
            "step": "verification"
        })

    return new_state

def create_migration_workflow(llm, mysql_conn_string: str, postgresql_conn_string: str) -> CompiledGraph:
    """Creates the workflow graph with LLM-based dependency resolution"""
    workflow = StateGraph(DBAgentState)

    # Create conversion chain
    conversion_chain = create_postgres_converter_chain(llm)

    # Add all nodes
    workflow.add_node("extract_schema", schema_extraction_node)
    workflow.add_node("convert_postgres", lambda x: postgres_converter_node(x, conversion_chain))
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("create_tables", lambda x: create_tables_node(x, postgresql_conn_string))
    workflow.add_node("extract_data", lambda x: extract_mysql_data_node(x, mysql_conn_string))
    workflow.add_node("solve_dependencies", solve_dependencies_node)
    workflow.add_node("load_data", lambda x: load_postgresql_data_node(x, postgresql_conn_string))
    workflow.add_node("verify_data", verification_node)

    # Add standard sequential edges
    workflow.add_edge("extract_schema", "convert_postgres")
    workflow.add_edge("convert_postgres", "human_review")
    workflow.add_edge("create_tables", "extract_data")
    workflow.add_edge("extract_data", "solve_dependencies")
    workflow.add_edge("solve_dependencies", "load_data")
    # workflow.add_edge("load_data", "verify_data")

    # Add conditional edges for review and dependency resolution
    workflow.add_conditional_edges(
        "human_review",
        lambda x: "convert_postgres" if x["review_status"] == "needs_revision" else "create_tables",
        {
            "convert_postgres": "convert_postgres",
            "create_tables": "create_tables"
        }
    )

    # Add conditional edge for dependency resolution retry
    workflow.add_conditional_edges(
        "load_data",
        lambda x: "solve_dependencies" if (
                x.get("errors") and
                any(e.get("step") == "data_loading" for e in x["errors"]) and
                x.get("dependency_attempts", 0) < 3- data analysis

        ) else "verify_data",
        {
            "solve_dependencies": "solve_dependencies",
            "verify_data": "verify_data"
        }
    )

    # Set entry and terminal nodes
    workflow.set_entry_point("extract_schema")
    workflow.set_finish_point("verify_data")

    return workflow.compile()


def initialize_state() -> DBAgentState:
    """Initialize the state with default values"""
    return {
        "messages": [],
        "mysql_schema": {},
        "postgresql_schema": "",
        "errors": [],
        "feedback": [],
        "review_status": "pending",
        "table_data": {},
        "row_counts": {},
        "dependency_attempts": 0,
        "table_order" : []
    }


async def run_conversion_workflow():
    """Run the complete workflow with both schema conversion and data migration"""
    log_message("[bold purple]=== Starting Database Migration Workflow ===[/bold purple]")
    log_message("[bold]Initializing workflow...[/bold]")
    ini = initialize_state()
    # Connection strings
    conn_mysql = "mysql+pymysql://root:deepak@localhost/employees"
    conn_postgres = ("postgresql://deepakachu:deepak@localhost:5432/employees")

    try:
        workflow = create_migration_workflow(llm, conn_mysql, conn_postgres)
        initial_state = initialize_state()

        log_message("\n[bold]Executing workflow...[/bold]")
        final_state = await workflow.ainvoke(initial_state)

        if final_state["review_status"] == "approved":
            log_message("\n[bold green]Workflow completed successfully![/bold green]")

            # Save the approved schema
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"postgresql_schema_{timestamp}.sql"

            with open(filename, "w") as f:
                f.write(final_state["postgresql_schema"])

            log_message(f"[green]Approved schema saved to: {filename}[/green]")

            # Print migration statistics
            log_message("\n[bold]Migration Statistics:[/bold]")
            for table, count in final_state["row_counts"].items():
                log_message(f"Table [cyan]{table}[/cyan]: {count:,} rows migrated")

        if final_state["errors"]:
            log_message("\n[bold red]Errors encountered during migration:[/bold red]")
            for error in final_state["errors"]:
                log_message(f"[red]- Step: {error.get('step', 'unknown')}")
                log_message(f"[red]- Error: {error.get('error', 'unknown error')}[/red]")

        return final_state, workflow.get_graph()

    except Exception as e:
        log_message(f"[bold red]Workflow failed with error: {str(e)}[/bold red]")
        raise


llm = ChatOpenAI(model="gpt-4o", api_key="")
db = SQLDatabase.from_uri("mysql+pymysql://root:deepak@localhost/employees")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

get_table_names = [item for item in tools if item.name == "sql_db_list_tables"][0]
get_schema = [item for item in tools if item.name == "sql_db_schema"][0]


if __name__ == "__main__":

    # Run the workflow
    sql, post, sname, pname = human_input()

    final_state, graph = asyncio.run(run_conversion_workflow())
    log_message("\n[bold]Workflow completed![/bold]")
    log_message(f"\n[bold]Successfully migrated SQL database : {sname}  to PostgreSQL database: {pname}[/bold]")
    # Generate and save workflow diagram
    log_message("\n[bold]Generating workflow diagram...[/bold]")
    try:
        pic = Image(
            graph.draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )

        display(pic)
        with open("schema_conversion_workflow.png", "wb") as f:
            f.write(pic.data)
        log_message("[green]✓ Workflow diagram saved as 'schema_conversion_workflow.png'[/green]")

    except Exception as e:
        log_message(f"[yellow]Warning: Could not generate workflow diagram: {str(e)}[/yellow]")



