"""Tests for Python 3.8 compatibility helpers."""

import pytest

from fibsem.utils import removesuffix, removeprefix


class TestRemoveSuffix:
    """Tests for removesuffix helper function."""

    def test_removesuffix_basic(self):
        """Test basic removesuffix functionality."""
        assert removesuffix("hello.txt", ".txt") == "hello"
        assert removesuffix("file.py", ".py") == "file"

    def test_removesuffix_no_match(self):
        """Test removesuffix when suffix doesn't match."""
        assert removesuffix("hello.txt", ".py") == "hello.txt"
        assert removesuffix("file", ".txt") == "file"

    def test_removesuffix_empty_suffix(self):
        """Test removesuffix with empty suffix."""
        assert removesuffix("hello", "") == "hello"

    def test_removesuffix_empty_string(self):
        """Test removesuffix with empty string."""
        assert removesuffix("", ".txt") == ""

    def test_removesuffix_longer_suffix(self):
        """Test removesuffix with longer suffix."""
        assert removesuffix("file.ome.tiff", ".ome.tiff") == "file"


class TestRemovePrefix:
    """Tests for removeprefix helper function."""

    def test_removeprefix_basic(self):
        """Test basic removeprefix functionality."""
        assert removeprefix("hello_world", "hello_") == "world"
        assert removeprefix("test_file", "test_") == "file"

    def test_removeprefix_no_match(self):
        """Test removeprefix when prefix doesn't match."""
        assert removeprefix("hello_world", "world_") == "hello_world"
        assert removeprefix("file", "test_") == "file"

    def test_removeprefix_empty_prefix(self):
        """Test removeprefix with empty prefix."""
        assert removeprefix("hello", "") == "hello"

    def test_removeprefix_empty_string(self):
        """Test removeprefix with empty string."""
        assert removeprefix("", "test_") == ""

    def test_removeprefix_longer_prefix(self):
        """Test removeprefix with longer prefix."""
        assert removeprefix("prefix_test_name", "prefix_") == "test_name"
