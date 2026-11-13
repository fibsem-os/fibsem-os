import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "named_loggers", ROOT / "scripts" / "codemods" / "named_loggers.py"
)
codemod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(codemod)


def test_adds_a_module_logger_after_the_imports():
    text = 'import logging\nimport os\n\nX = 1\n\n\ndef f():\n    logging.warn("a %s", os.sep)\n'
    new, _ = codemod.convert(text)
    assert new == (
        "import logging\nimport os\n\nlogger = logging.getLogger(__name__)\n"
        '\nX = 1\n\n\ndef f():\n    logger.warning("a %s", os.sep)\n'
    )
    assert codemod.convert(new) == (None, "")


def test_keeps_an_existing_module_logger():
    text = 'import logging\n\nlog = logging.getLogger(__name__)\n\nlogging.info("é"); logging.error("x")\n'
    new, _ = codemod.convert(text)
    assert new.endswith('log.info("é"); log.error("x")\n')
    assert new.count("getLogger") == 1


def test_logger_goes_above_an_import_block_that_logs():
    text = (
        "import logging\n\ntry:\n    import x\nexcept ImportError:\n"
        '    logging.debug("no x")\n'
    )
    new, _ = codemod.convert(text)
    assert new.index("logger = logging.getLogger") < new.index("try:")
    assert 'logger.debug("no x")' in new


def test_another_name_when_logger_is_taken():
    text = 'import logging\n\n\ndef f(logger):\n    logging.info("a")\n'
    new, _ = codemod.convert(text)
    assert "_logger = logging.getLogger(__name__)" in new
    assert '_logger.info("a")' in new


def test_leaves_a_module_that_rebinds_logging():
    text = 'from x import logging\n\nlogging.info("a")\n'
    new, why = codemod.convert(text)
    assert new is None and "stdlib" in why
