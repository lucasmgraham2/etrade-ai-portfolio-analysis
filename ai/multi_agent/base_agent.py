"""
Base Agent Class
Provides common functionality for all specialized agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime
import json


class BaseAgent(ABC):
    """Abstract base class for all portfolio analysis agents"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the base agent
        
        Args:
            name: Agent name
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.results = {}
        self.execution_time = None
        self.status = "initialized"
        self.errors = []
        
    @abstractmethod
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method - must be implemented by each agent
        
        Args:
            context: Analysis context including portfolio data and other agent outputs
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent analysis with error handling and timing
        
        Args:
            context: Analysis context
            
        Returns:
            Agent execution results including metadata
        """
        start_time = datetime.now()
        self.status = "running"
        
        try:
            print(f"🤖 {self.name} starting analysis...")
            self.results = await self.analyze(context)
            self.status = "completed"
            print(f"✓ {self.name} completed successfully")
            
        except Exception as e:
            self.status = "failed"
            error_msg = f"Error in {self.name}: {str(e)}"
            self.errors.append(error_msg)
            print(f"✗ {self.name} failed: {str(e)}")
            self.results = {"error": str(e)}
            
        finally:
            end_time = datetime.now()
            self.execution_time = (end_time - start_time).total_seconds()
            
        return self.get_output()
    
    def get_output(self) -> Dict[str, Any]:
        """
        Get formatted agent output
        
        Returns:
            Dictionary containing results and metadata
        """
        return {
            "agent": self.name,
            "status": self.status,
            "execution_time": self.execution_time,
            "results": self.results,
            "errors": self.errors,
            "timestamp": datetime.now().isoformat()
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{self.name}] [{level}] {message}")
        
    def validate_context(self, context: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that required keys are present in context
        
        Args:
            context: Context dictionary to validate
            required_keys: List of required key names
            
        Returns:
            True if valid, False otherwise
        """
        missing_keys = [key for key in required_keys if key not in context]
        if missing_keys:
            error_msg = f"Missing required context keys: {', '.join(missing_keys)}"
            self.errors.append(error_msg)
            self.log(error_msg, "ERROR")
            return False
        return True
