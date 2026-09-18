"""CLI argument parsing tests."""

from image_translation.input.arguments import parse_arguments


def test_output_folder_argument():
    parsed = parse_arguments(["photos", "-o", "D:/out/localized"])
    assert parsed.output_folder.name == "localized"
