# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Filter parsing and wildcard matching."""
from __future__ import annotations

import pytest

from canlens.filters import (
    TraceFilter,
    bus_names,
    can_id_names,
    compile_pattern,
    matches,
    segment_names,
)


class TestWildcards:
    def test_star_matches_any_run_including_nothing(self):
        assert matches("0x1*", "0x1")
        assert matches("0x1*", "0x11")
        assert matches("0x1*", "0x123")

    def test_question_mark_matches_exactly_one_character(self):
        assert matches("0x1?", "0x11")
        assert not matches("0x1?", "0x1")
        assert not matches("0x1?", "0x123")

    def test_the_documented_example(self):
        # "0x1* -> 0x1, 0x11, 0x123, 0x124, 0x112 and so on"
        kept = [a for a in (0x1, 0x11, 0x123, 0x124, 0x112, 0x2, 0x211, 0x21)
                if matches("0x1*", *can_id_names(a))]
        assert kept == [0x1, 0x11, 0x123, 0x124, 0x112]

    def test_patterns_are_anchored_at_both_ends(self):
        assert not matches("0x1", "0x12")
        assert not matches("x1", "0x1")

    def test_matching_ignores_case(self):
        assert matches("0X1a*", "0x1A5")
        assert matches("toyota*", "TOYOTA_PRIUS")

    def test_brackets_are_literal_not_a_character_class(self):
        # A glob library would read [ab] as a class; only * and ? are special.
        assert matches("a[ab]c", "a[ab]c")
        assert not matches("a[ab]c", "aac")

    def test_other_regex_metacharacters_are_literal(self):
        assert matches("a.c", "a.c")
        assert not matches("a.c", "abc")
        assert matches("a+b", "a+b")

    def test_star_alone_and_empty_keep_everything(self):
        assert matches("*", "anything")
        assert matches("", "anything")
        assert matches("   ", "anything")

    def test_compile_pattern_is_reusable(self):
        regex = compile_pattern("0x2??")
        assert regex.fullmatch("0x211") and not regex.fullmatch("0x21")


class TestRenderings:
    @pytest.mark.parametrize("text", ["CAN0", "CAN_0", "bus 0", "bus0", "0"])
    def test_a_bus_can_be_written_several_ways(self, text):
        assert matches(text, *bus_names(0))

    def test_bus_patterns_do_not_leak_between_buses(self):
        assert not matches("CAN0", *bus_names(1))
        assert not matches("bus 2", *bus_names(0))

    def test_identifiers_match_with_or_without_the_prefix(self):
        assert matches("0x211", *can_id_names(0x211))
        assert matches("211", *can_id_names(0x211))

    def test_identifiers_carry_no_leading_zeroes(self):
        # Padding would break 0x1* as a prefix filter.
        assert can_id_names(0x1) == ("0x1", "1")

    def test_segment_can_be_named_by_route_or_index(self):
        names = segment_names("/data/segments/dev0/00000044--abc/464/rlog.zst")
        assert "dev0/00000044--abc/464" in names
        assert "00000044--abc/464" in names
        assert "00000044" in names
        assert "464" in names


class TestParsing:
    def test_the_documented_filter(self):
        f = TraceFilter.parse("TOYOTA_PRIUS,*,CAN0,*")
        assert (f.vehicle, f.segment, f.bus, f.can_id) == ("TOYOTA_PRIUS", "*", "CAN0", "*")

    def test_missing_trailing_fields_default_to_everything(self):
        assert TraceFilter.parse("TOYOTA_PRIUS") == TraceFilter("TOYOTA_PRIUS", "*", "*", "*")
        assert TraceFilter.parse("KIA_EV6,*") == TraceFilter("KIA_EV6", "*", "*", "*")

    def test_semicolons_work_as_well_as_commas(self):
        assert TraceFilter.parse("A;B;C;D") == TraceFilter("A", "B", "C", "D")

    def test_whitespace_around_fields_is_ignored(self):
        assert TraceFilter.parse(" A , B , C , D ") == TraceFilter("A", "B", "C", "D")

    def test_an_empty_field_means_everything(self):
        assert TraceFilter.parse("A,,C,").segment == "*"

    def test_empty_input_keeps_everything(self):
        assert TraceFilter.parse("").is_empty
        assert TraceFilter.parse("   ").is_empty
        assert TraceFilter().is_empty

    def test_too_many_fields_is_rejected_with_a_useful_message(self):
        with pytest.raises(ValueError, match="at most 4 fields"):
            TraceFilter.parse("a,b,c,d,e")

    def test_round_trips_through_its_own_text(self):
        f = TraceFilter.parse("TOYOTA_PRIUS,*,CAN0,0x2*")
        assert TraceFilter.parse(str(f)) == f


class TestSelection:
    def test_the_documented_filter_selects_a_vehicle_and_a_bus(self):
        f = TraceFilter.parse("TOYOTA_PRIUS,*,CAN0,*")
        assert f.matches_segment("TOYOTA_PRIUS", "d/r/1")
        assert not f.matches_segment("KIA_EV6", "d/r/1")
        assert f.matches_message(0, 0x211)
        assert not f.matches_message(1, 0x211)

    def test_identifier_and_bus_are_both_required(self):
        f = TraceFilter.parse("*,*,CAN1,0x2*")
        assert f.matches_message(1, 0x211)
        assert not f.matches_message(0, 0x211)
        assert not f.matches_message(1, 0x111)

    def test_an_empty_filter_keeps_everything(self):
        f = TraceFilter()
        assert f.matches_segment(None, "d/r/1")
        assert f.matches_message(3, 0x18DAF110)

    def test_selects_segments_reports_whether_the_list_narrows(self):
        assert not TraceFilter.parse("*,*,CAN0,0x1*").selects_segments
        assert TraceFilter.parse("TOYOTA_PRIUS").selects_segments
        assert TraceFilter.parse("*,00000044").selects_segments

    def test_a_missing_platform_only_matches_a_wildcard(self):
        assert TraceFilter.parse("*").matches_segment(None, "d/r/1")
        assert not TraceFilter.parse("TOYOTA_PRIUS").matches_segment(None, "d/r/1")

    def test_extended_identifiers_filter_by_prefix(self):
        f = TraceFilter.parse("*,*,*,0x18DA*")
        assert f.matches_message(2, 0x18DAF110)
        assert not f.matches_message(2, 0x18DB33F1)
