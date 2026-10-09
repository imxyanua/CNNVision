import unittest

from app_i18n import LANG_LABELS, TEXT, translate


class I18nTest(unittest.TestCase):
    def test_three_languages_share_keys(self) -> None:
        self.assertEqual(set(LANG_LABELS), {"vi", "zh", "en"})
        keys = set(TEXT["vi"])
        self.assertEqual(set(TEXT["zh"]), keys)
        self.assertEqual(set(TEXT["en"]), keys)

    def test_translate_falls_back_to_vietnamese(self) -> None:
        self.assertEqual(translate("nope", "upload"), TEXT["vi"]["upload"])

    def test_unknown_key_returns_the_key(self) -> None:
        self.assertEqual(translate("en", "not_a_real_key"), "not_a_real_key")
