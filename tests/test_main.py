"""
Test file for main.py
"""
import unittest
import sys
import os

# Add the parent directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main

class TestMain(unittest.TestCase):
    """Test cases for the main module."""
    
    def test_main_function_exists(self):
        """Test that the main function exists and is callable."""
        self.assertTrue(callable(main))

if __name__ == "__main__":
    unittest.main()