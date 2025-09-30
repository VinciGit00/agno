"""WorkflowAgent - A restricted Agent for workflow orchestration"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from agno.agent import Agent
from agno.models.base import Model
from agno.utils.log import logger, log_info, log_debug

if TYPE_CHECKING:
    from agno.session.workflow import WorkflowSession
    from agno.workflow.types import WorkflowExecutionInput


class WorkflowAgent(Agent):
    """
    A restricted Agent class specifically designed for workflow orchestration.

    This agent can:
    1. Decide whether to run the workflow or answer directly from history
    2. Call the workflow execution tool when needed
    3. Access workflow session history for context

    Restrictions:
    - Only model configuration allowed
    - No custom tools (tools are set by workflow)
    - No knowledge base
    - Limited configuration options
    """

    def __init__(
        self,
        model: Model,
        instructions: Optional[str] = None,
    ):
        """
        Initialize WorkflowAgent with restricted parameters.

        Args:
            model: The model to use for the agent (required)
            name: Agent name (defaults to "Workflow Agent")
            description: Agent description
            instructions: Custom instructions (will be combined with workflow context)
        """

        super().__init__(
            model=model,
            instructions=instructions,
        )

    def create_workflow_tool(
        self,
        workflow: "Any",  # Workflow type
        session: "WorkflowSession",
        execution_input: "WorkflowExecutionInput",
        session_state: Optional[Dict[str, Any]],
        stream_intermediate_steps: bool = False,
    ) -> Callable:
        """
        Create the workflow execution tool that this agent can call.

        This is similar to how Agent has search_knowledge_base() method.

        Args:
            workflow: The workflow instance
            session: The workflow session
            execution_input: The execution input
            session_state: The session state

        Returns:
            Callable tool function
        """
        from datetime import datetime
        from uuid import uuid4

        from pydantic import BaseModel

        from agno.run.workflow import WorkflowRunOutput
        from agno.utils.log import log_debug
        from agno.workflow.types import WorkflowExecutionInput

        def run_workflow(query: str):
            """
            Execute the complete workflow with the given query.
            Use this tool when you need to run the workflow to answer the user's question.

            Args:
                query: The input query/question to process through the workflow

            Returns:
                The workflow execution result (str in non-streaming, generator in streaming)
            """
            # STREAMING MODE: Return a generator that yields workflow events
            if stream_intermediate_steps:
                logger.info("=" * 80)
                logger.info("⚙️ TOOL EXECUTION (STREAMING): run_workflow")
                logger.info("=" * 80)
                
                # Create a new run ID for this execution
                run_id = str(uuid4())
                log_debug(f"🆔 Created new run ID: {run_id}")

                # Create workflow run response
                workflow_run_response = WorkflowRunOutput(
                    run_id=run_id,
                    input=query,
                    session_id=session.session_id,
                    workflow_id=workflow.id,
                    workflow_name=workflow.name,
                    created_at=int(datetime.now().timestamp()),
                )

                # Update the execution input with the agent's refined query
                workflow_execution_input = WorkflowExecutionInput(
                    input=query,
                    additional_data=execution_input.additional_data,
                    audio=execution_input.audio,
                    images=execution_input.images,
                    videos=execution_input.videos,
                    files=execution_input.files,
                )
                
                log_debug("🚀 Executing workflow with streaming...")

                # Execute workflow with streaming and yield all events
                final_content = ""
                for event in workflow._execute_stream(
                    session=session,
                    execution_input=workflow_execution_input,
                    workflow_run_response=workflow_run_response,
                    session_state=session_state,
                    stream_intermediate_steps=True,  # Stream all workflow events
                ):
                    # Yield workflow events to bubble them up through the agent
                    yield event
                    
                    # Capture final content from WorkflowCompletedEvent
                    from agno.run.workflow import WorkflowCompletedEvent
                    if isinstance(event, WorkflowCompletedEvent):
                        final_content = str(event.content) if event.content else ""
                
                logger.info("=" * 80)
                logger.info("✅ TOOL EXECUTION COMPLETE (STREAMING): run_workflow")
                logger.info("=" * 80)
                
                # The final return value becomes the tool's output
                return final_content

            # NON-STREAMING MODE: Execute synchronously
            logger.info("=" * 80)
            logger.info("⚙️ TOOL EXECUTION: run_workflow")
            logger.info("=" * 80)

            # Create a new run ID for this execution
            run_id = str(uuid4())
            log_debug(f"🆔 Created new run ID: {run_id}")

            # Create workflow run response
            workflow_run_response = WorkflowRunOutput(
                run_id=run_id,
                input=query,
                session_id=session.session_id,
                workflow_id=workflow.id,
                workflow_name=workflow.name,
                created_at=int(datetime.now().timestamp()),
            )

            # Update the execution input with the agent's refined query
            workflow_execution_input = WorkflowExecutionInput(
                input=query,
                additional_data=execution_input.additional_data,
                audio=execution_input.audio,
                images=execution_input.images,
                videos=execution_input.videos,
                files=execution_input.files,
            )

            # Execute the workflow (non-streaming)
            log_debug("🚀 Executing workflow steps...")
            result = workflow._execute(
                session=session,
                execution_input=workflow_execution_input,
                workflow_run_response=workflow_run_response,
                session_state=session_state,
            )

            logger.info("=" * 80)
            logger.info("✅ TOOL EXECUTION COMPLETE: run_workflow")
            logger.info(f"    ➜ Run ID: {result.run_id}")
            logger.info(f"    ➜ Result length: {len(str(result.content)) if result.content else 0} chars")
            logger.info("=" * 80)

            # Return the content as string
            if isinstance(result.content, str):
                return result.content
            elif isinstance(result.content, BaseModel):
                return result.content.model_dump_json(exclude_none=True)
            else:
                return str(result.content)

        return run_workflow
