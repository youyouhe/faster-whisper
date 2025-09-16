#!/usr/bin/env python3
"""
SRT subtitle merging and timestamp alignment utilities
"""

import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SubtitleSegment:
    """Represents a subtitle segment with timing and text"""
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    chunk_index: int   # which audio chunk this came from

class SRTMerger:
    """Merge and align SRT subtitles from multiple audio chunks"""

    def __init__(self, overlap_seconds: float = 2.0):
        self.overlap_seconds = overlap_seconds

    def parse_srt_content(self, srt_content: str, chunk_index: int, chunk_start_time: float) -> List[SubtitleSegment]:
        """
        Parse SRT content into subtitle segments with absolute timestamps

        Args:
            srt_content: SRT content string
            chunk_index: Index of the audio chunk
            chunk_start_time: Start time of this chunk in the original audio

        Returns:
            List of SubtitleSegment objects
        """
        segments = []

        # SRT format: index, start_time --> end_time, text, empty line
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\d+|\Z)'
        matches = re.findall(pattern, srt_content.strip())

        for match in matches:
            index, start_time_str, end_time_str, text = match

            # Convert SRT time format to seconds
            start_time = self._srt_time_to_seconds(start_time_str)
            end_time = self._srt_time_to_seconds(end_time_str)

            # Adjust timestamp to absolute position in original audio
            absolute_start = start_time + chunk_start_time
            absolute_end = end_time + chunk_start_time

            segment = SubtitleSegment(
                start_time=absolute_start,
                end_time=absolute_end,
                text=text.strip(),
                chunk_index=chunk_index
            )
            segments.append(segment)

        logger.info(f"Parsed {len(segments)} segments from chunk {chunk_index}")
        return segments

    def _srt_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT time format HH:MM:SS,mmm to seconds"""
        time_part, ms_part = time_str.split(',')
        hours, minutes, seconds = time_part.split(':')
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms_part) / 1000

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_part = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"

    def merge_segments(self, all_segments: List[List[SubtitleSegment]], chunk_timings: List[Tuple[float, float]]) -> List[SubtitleSegment]:
        """
        Merge segments from all chunks, removing overlaps and sorting by time

        Args:
            all_segments: List of segments from each chunk
            chunk_timings: List of (start_time, end_time) for each chunk

        Returns:
            List of merged and sorted SubtitleSegment objects
        """
        logger.info(f"Merging {len(all_segments)} chunk segments")

        # Flatten all segments
        flattened = []
        for chunk_idx, segments in enumerate(all_segments):
            chunk_start_time, _ = chunk_timings[chunk_idx]
            for segment in segments:
                # Adjust segment timing to account for overlap removal
                adjusted_segment = self._adjust_segment_timing(segment, chunk_idx, chunk_timings)
                flattened.append(adjusted_segment)

        # Sort by start time
        flattened.sort(key=lambda x: x.start_time)

        # Remove overlaps
        merged_segments = self._remove_overlaps(flattened)

        logger.info(f"Merged to {len(merged_segments)} segments")
        return merged_segments

    def _adjust_segment_timing(self, segment: SubtitleSegment, chunk_index: int, chunk_timings: List[Tuple[float, float]]) -> SubtitleSegment:
        """
        Adjust segment timing to remove overlap regions

        Args:
            segment: Original segment
            chunk_index: Index of the chunk
            chunk_timings: Timing information for all chunks

        Returns:
            Adjusted SubtitleSegment
        """
        chunk_start, chunk_end = chunk_timings[chunk_index]

        # For chunks after the first one, remove overlap from the beginning
        if chunk_index > 0:
            # The first overlap_seconds should be removed from the start
            adjusted_start = max(segment.start_time, chunk_start + self.overlap_seconds)
            adjusted_end = segment.end_time
        else:
            adjusted_start = segment.start_time
            adjusted_end = segment.end_time

        return SubtitleSegment(
            start_time=adjusted_start,
            end_time=adjusted_end,
            text=segment.text,
            chunk_index=segment.chunk_index
        )

    def _remove_overlaps(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """
        Remove overlapping segments by merging or truncating

        Args:
            segments: List of segments sorted by start time

        Returns:
            List of non-overlapping segments
        """
        if not segments:
            return []

        merged = [segments[0]]

        for current in segments[1:]:
            previous = merged[-1]

            # Check for overlap
            if current.start_time < previous.end_time:
                # Overlap detected, merge segments
                merged_text = f"{previous.text} {current.text}"
                merged_segment = SubtitleSegment(
                    start_time=previous.start_time,
                    end_time=max(previous.end_time, current.end_time),
                    text=merged_text.strip(),
                    chunk_index=previous.chunk_index
                )
                merged[-1] = merged_segment
            else:
                # No overlap, add as is
                merged.append(current)

        return merged

    def generate_srt_content(self, segments: List[SubtitleSegment]) -> str:
        """
        Generate final SRT content from merged segments

        Args:
            segments: List of merged SubtitleSegment objects

        Returns:
            SRT content as string
        """
        srt_lines = []
        for i, segment in enumerate(segments, 1):
            start_time_str = self._seconds_to_srt_time(segment.start_time)
            end_time_str = self._seconds_to_srt_time(segment.end_time)

            srt_lines.append(str(i))
            srt_lines.append(f"{start_time_str} --> {end_time_str}")
            srt_lines.append(segment.text)
            srt_lines.append("")  # Empty line between segments

        return "\n".join(srt_lines)

    def merge_chunk_results(self, chunk_results: List[str], chunk_timings: List[Tuple[float, float]]) -> str:
        """
        Main method to merge SRT results from multiple chunks

        Args:
            chunk_results: List of SRT content strings from each chunk
            chunk_timings: List of (start_time, end_time) for each chunk

        Returns:
            Final merged SRT content
        """
        logger.info(f"Merging {len(chunk_results)} chunk results")

        # Parse all chunk results
        all_segments = []
        for chunk_idx, (srt_content, (start_time, _)) in enumerate(zip(chunk_results, chunk_timings)):
            segments = self.parse_srt_content(srt_content, chunk_idx, start_time)
            all_segments.append(segments)

        # Merge segments
        merged_segments = self.merge_segments(all_segments, chunk_timings)

        # Generate final SRT
        final_srt = self.generate_srt_content(merged_segments)

        return final_srt