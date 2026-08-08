#!/usr/bin/env python3
"""
Test script for SRT merging improvements (standalone)
"""

import re
import logging
from typing import List, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import unicodedata

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SubtitleSegment:
    """Represents a subtitle segment with timing and text"""
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    chunk_index: int   # which audio chunk this came from
    confidence: float = 1.0  # transcription confidence (if available)

    def duration(self) -> float:
        """Return segment duration"""
        return max(0, self.end_time - self.start_time)

    def is_valid(self) -> bool:
        """Check if segment has valid timing"""
        return self.start_time >= 0 and self.end_time > self.start_time

class SRTMerger:
    """Merge and align SRT subtitles from multiple audio chunks"""

    def __init__(self, overlap_seconds: float = 2.0):
        self.overlap_seconds = overlap_seconds

    def _normalize_text(self, text: str) -> str:
        """Normalize text for similarity comparison"""
        # Remove punctuation and convert to lowercase
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.category(c).startswith('P'))
        text = text.lower().strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio"""
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        return SequenceMatcher(None, norm1, norm2).ratio()

    def _find_best_text_in_overlap(self, segment1: SubtitleSegment, segment2: SubtitleSegment) -> Tuple[str, float]:
        """
        Choose the best text representation for overlapping segments

        Returns:
            Tuple of (chosen_text, confidence_score)
        """
        # Calculate text similarity
        similarity = self._text_similarity(segment1.text, segment2.text)

        # Check for exact duplicates or very high similarity
        if similarity > 0.9:
            # Very similar, choose the longer, more complete text
            if len(segment1.text) > len(segment2.text):
                return segment1.text, 1.0
            else:
                return segment2.text, 1.0

        # If texts are somewhat similar but not identical, try to combine intelligently
        elif similarity > 0.6:
            # Try to merge text without duplication
            merged_text = self._merge_texts_without_duplication(segment1.text, segment2.text)
            if merged_text:
                return merged_text, 0.9
            else:
                # If merging failed, choose longer text
                longer_text = segment1.text if len(segment1.text) >= len(segment2.text) else segment2.text
                return longer_text, 0.8

        # If overlap is significant (>50% of either segment), this might be a problem
        overlap_start = max(segment1.start_time, segment2.start_time)
        overlap_end = min(segment1.end_time, segment2.end_time)
        overlap_duration = max(0, overlap_end - overlap_start)

        if overlap_duration > 0:
            overlap_ratio1 = overlap_duration / segment1.duration()
            overlap_ratio2 = overlap_duration / segment2.duration()

            if overlap_ratio1 > 0.5 or overlap_ratio2 > 0.5:
                # Significant overlap, choose longer segment
                if segment1.duration() >= segment2.duration():
                    return segment1.text, 0.7
                else:
                    return segment2.text, 0.7

        # Minor overlap or no overlap, prefer segment from earlier chunk
        if segment1.chunk_index <= segment2.chunk_index:
            return segment1.text, 0.9
        else:
            return segment2.text, 0.9

    def _merge_texts_without_duplication(self, text1: str, text2: str) -> str:
        """
        Try to merge two texts while removing duplicate parts

        Returns:
            Merged text without duplicates, or None if merging failed
        """
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)

        # Find common prefix and suffix
        common_prefix = ""
        for i in range(min(len(norm1), len(norm2))):
            if norm1[i] == norm2[i]:
                common_prefix += norm1[i]
            else:
                break

        # Try to find common suffix
        common_suffix = ""
        norm1_rev = norm1[::-1]
        norm2_rev = norm2[::-1]
        for i in range(min(len(norm1_rev), len(norm2_rev))):
            if norm1_rev[i] == norm2_rev[i]:
                common_suffix = norm1_rev[i] + common_suffix
            else:
                break

        # If we found significant overlap, try to construct merged text
        if len(common_prefix) > 5 or len(common_suffix) > 5:
            # Extract the unique parts
            unique1 = text1[len(common_prefix):] if len(common_prefix) > 0 else text1
            unique2 = text2[:len(text2) - len(common_suffix)] if len(common_suffix) > 0 else text2

            # Remove common suffix from unique1 if present
            if common_suffix and unique1.endswith(common_suffix):
                unique1 = unique1[:-len(common_suffix)]

            # Construct merged text
            merged = text1 if len(text1) >= len(text2) else text2

            # Try to add unique parts
            if unique1.strip() and unique1 not in merged:
                merged = merged + " " + unique1.strip()
            if unique2.strip() and unique2 not in merged:
                merged = unique2.strip() + " " + merged

            return merged.strip()

        return None

    def _remove_overlaps_intelligent(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """
        Remove overlapping segments using intelligent merging based on text similarity and timing

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
                # Overlap detected
                overlap_duration = previous.end_time - current.start_time

                logger.debug(f"Overlap detected: previous({previous.start_time:.3f}-{previous.end_time:.3f}) "
                           f"current({current.start_time:.3f}-{current.end_time:.3f}) "
                           f"overlap={overlap_duration:.3f}s")

                # Choose the best text representation
                chosen_text, confidence = self._find_best_text_in_overlap(previous, current)

                # Handle the overlap based on overlap duration and text similarity
                if overlap_duration > 1.0 or confidence < 0.9:
                    # Significant overlap or low confidence - merge segments
                    merged_text = chosen_text
                    merged_segment = SubtitleSegment(
                        start_time=previous.start_time,
                        end_time=max(previous.end_time, current.end_time),
                        text=merged_text.strip(),
                        chunk_index=min(previous.chunk_index, current.chunk_index),
                        confidence=confidence
                    )
                    merged[-1] = merged_segment

                    logger.debug(f"Merged overlapping segments: '{merged_text[:50]}...'")

                else:
                    # Minor overlap, adjust timing
                    # Extend the previous segment if current is longer and starts soon after
                    if current.end_time > previous.end_time:
                        # Truncate current to start after previous ends
                        adjusted_current = SubtitleSegment(
                            start_time=previous.end_time + 0.1,  # Small gap
                            end_time=current.end_time,
                            text=current.text,
                            chunk_index=current.chunk_index,
                            confidence=current.confidence * 0.9  # Slightly reduce confidence
                        )

                        # Only add if the adjusted segment is meaningful
                        if adjusted_current.duration() > 0.5:
                            merged.append(adjusted_current)
                            logger.debug(f"Adjusted overlapping segment: {adjusted_current.start_time:.3f}-{adjusted_current.end_time:.3f}")
                        else:
                            logger.debug(f"Skipping very short adjusted segment: {adjusted_current.duration():.3f}s")
            else:
                # No overlap, add as is
                merged.append(current)

        # Final validation and cleanup
        final_segments = []
        for segment in merged:
            if segment.is_valid() and segment.duration() > 0.1:
                # Ensure minimum gap between segments
                if final_segments and segment.start_time - final_segments[-1].end_time < 0.05:
                    # Merge with previous if gap is too small
                    prev = final_segments[-1]
                    merged_segment = SubtitleSegment(
                        start_time=prev.start_time,
                        end_time=segment.end_time,
                        text=prev.text + " " + segment.text,
                        chunk_index=prev.chunk_index,
                        confidence=min(prev.confidence, segment.confidence)
                    )
                    final_segments[-1] = merged_segment
                else:
                    final_segments.append(segment)

        return final_segments

    def merge_segments(self, all_segments: List[List[SubtitleSegment]], chunk_timings: List[Tuple[float, float, float, float]]) -> List[SubtitleSegment]:
        """
        Merge segments from all chunks, removing overlaps and sorting by time

        Args:
            all_segments: List of segments from each chunk
            chunk_timings: List of (actual_start, actual_end, theoretical_start, theoretical_end) for each chunk

        Returns:
            List of merged and sorted SubtitleSegment objects
        """
        logger.info(f"Merging {len(all_segments)} chunk segments")

        # Flatten all segments with validation
        flattened = []
        for chunk_idx, segments in enumerate(all_segments):
            logger.info(f"Processing {len(segments)} segments from chunk {chunk_idx}")
            for segment in segments:
                if segment.is_valid():
                    flattened.append(segment)
                else:
                    logger.warning(f"Skipping invalid segment from chunk {chunk_idx}: "
                                 f"start={segment.start_time:.3f}, end={segment.end_time:.3f}")

        if not flattened:
            logger.warning("No valid segments to merge")
            return []

        # Sort by start time and chunk index for stable ordering
        flattened.sort(key=lambda x: (x.start_time, x.chunk_index))

        logger.info(f"After sorting: {len(flattened)} segments")
        for i, seg in enumerate(flattened[:5]):  # Log first few for debugging
            logger.info(f"  {i}: Chunk {seg.chunk_index}, {seg.start_time:.3f}-{seg.end_time:.3f}: {seg.text[:50]}...")

        # Remove overlaps with intelligent merging
        merged_segments = self._remove_overlaps_intelligent(flattened)

        logger.info(f"After overlap removal: {len(merged_segments)} segments")
        return merged_segments

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

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_part = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"

    def merge_chunk_results(self, chunk_results: List[str], chunk_timings: List[Tuple[float, float, float, float]]) -> str:
        """
        Main method to merge SRT results from multiple chunks

        Args:
            chunk_results: List of SRT content strings from each chunk
            chunk_timings: List of (actual_start, actual_end, theoretical_start, theoretical_end) for each chunk

        Returns:
            Final merged SRT content
        """
        logger.info(f"Merging {len(chunk_results)} chunk results")

        if len(chunk_results) != len(chunk_timings):
            raise ValueError(f"Number of results ({len(chunk_results)}) doesn't match number of timings ({len(chunk_timings)})")

        # Parse all chunk results
        all_segments = []
        for chunk_idx, (srt_content, timing) in enumerate(zip(chunk_results, chunk_timings)):
            segments = self.parse_srt_content(srt_content, chunk_idx, timing)
            all_segments.append(segments)

        # Merge segments
        merged_segments = self.merge_segments(all_segments, chunk_timings)

        # Generate final SRT
        final_srt = self.generate_srt_content(merged_segments)

        logger.info(f"Generated final SRT with {len(merged_segments)} segments")
        return final_srt

    def parse_srt_content(self, srt_content: str, chunk_index: int, chunk_timing: Tuple[float, float, float, float]) -> List[SubtitleSegment]:
        """
        Parse SRT content into subtitle segments with absolute timestamps

        Args:
            srt_content: SRT content string
            chunk_index: Index of the audio chunk
            chunk_timing: (actual_start, actual_end, theoretical_start, theoretical_end)

        Returns:
            List of SubtitleSegment objects
        """
        actual_start, actual_end, theoretical_start, theoretical_end = chunk_timing
        segments = []

        if not srt_content or not srt_content.strip():
            logger.warning(f"Empty SRT content for chunk {chunk_index}")
            return segments

        try:
            # SRT format: index, start_time --> end_time, text, empty line
            pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\d+|\Z)'
            matches = re.findall(pattern, srt_content.strip())

            logger.info(f"Found {len(matches)} SRT segments in chunk {chunk_index}")

            for match in matches:
                index, start_time_str, end_time_str, text = match

                try:
                    # Convert SRT time format to seconds
                    relative_start = self._srt_time_to_seconds(start_time_str)
                    relative_end = self._srt_time_to_seconds(end_time_str)

                    # Calculate offset based on actual vs theoretical timing
                    # The relative timing is based on actual chunk start
                    offset = actual_start - theoretical_start
                    absolute_start = relative_start + offset
                    absolute_end = relative_end + offset

                    # Create segment
                    segment = SubtitleSegment(
                        start_time=absolute_start,
                        end_time=absolute_end,
                        text=text.strip(),
                        chunk_index=chunk_index
                    )

                    # Validate segment
                    if segment.is_valid() and segment.duration() > 0.1:  # At least 0.1 second
                        segments.append(segment)
                    else:
                        logger.warning(f"Invalid segment {index} in chunk {chunk_index}: "
                                     f"start={absolute_start:.3f}, end={absolute_end:.3f}")

                except Exception as e:
                    logger.error(f"Error parsing SRT segment {index} in chunk {chunk_index}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing SRT content for chunk {chunk_index}: {e}")

        logger.info(f"Successfully parsed {len(segments)} valid segments from chunk {chunk_index}")
        return segments

    def _srt_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT time format HH:MM:SS,mmm to seconds"""
        time_part, ms_part = time_str.split(',')
        hours, minutes, seconds = time_part.split(':')
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms_part) / 1000

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_part = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"

def create_mock_srt_content(start_offset: float, segments_data: list) -> str:
    """Create mock SRT content for testing"""
    lines = []
    for i, (start_time, end_time, text) in enumerate(segments_data):
        # Add offset to simulate chunk timing
        actual_start = start_offset + start_time
        actual_end = start_offset + end_time

        # Convert to SRT format
        start_srt = seconds_to_srt_time(actual_start)
        end_srt = seconds_to_srt_time(actual_end)

        lines.append(f"{i+1}")
        lines.append(f"{start_srt} --> {end_srt}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)

def test_srt_merging():
    """Test the improved SRT merging logic"""
    logger.info("=== Testing SRT Merging Improvements ===")

    # Create test data simulating overlapping chunks
    merger = SRTMerger(overlap_seconds=2.0)

    # Mock chunk timings: (actual_start, actual_end, theoretical_start, theoretical_end)
    chunk_timings = [
        (0.0, 12.0, 0.0, 10.0),      # Chunk 0: 0-10s theoretical, 0-12s actual (2s overlap at end)
        (8.0, 22.0, 10.0, 20.0),     # Chunk 1: 10-20s theoretical, 8-22s actual (2s overlap both sides)
        (18.0, 30.0, 20.0, 30.0)     # Chunk 2: 20-30s theoretical, 18-30s actual (2s overlap at start)
    ]

    # Create mock SRT contents with overlapping and problematic content (simulating your original problem)
    chunk_results = [
        # Chunk 0: Original content
        create_mock_srt_content(0.0, [
            (0.5, 2.0, "这是第一段话的开始"),
            (2.5, 4.0, "这是正常的中间内容"),
            (4.5, 6.0, "那时签下的是功名利禄"),
            (8.0, 10.0, "却不知那是一纸催命符")  # This will overlap with chunk 1
        ]),

        # Chunk 1: Overlapping content with problematic timing (simulating the original error)
        create_mock_srt_content(10.0, [
            (0.0, 2.0, "却不知那是一纸催命符"),    # Duplicate from chunk 0
            (1.5, 3.0, "酒醒之后我才看清了"),     # Overlapping with previous
            (4.0, 6.0, "那要命的真相"),          # Overlapping
            (8.0, 10.0, "赤令上的一枚贴黄")       # This will overlap with chunk 2
        ]),

        # Chunk 2: More overlapping content
        create_mock_srt_content(20.0, [
            (0.0, 1.5, "赤令上的一枚贴黄"),       # Duplicate from chunk 1
            (2.0, 4.0, "不知何时脱落"),          # Normal content
            (5.0, 7.0, "露出的不是尖"),          # Normal content
            (8.0, 9.5, "而是一个清晰的仙字")      # Normal content
        ])
    ]

    logger.info("Created test data:")
    for i, (content, timing) in enumerate(zip(chunk_results, chunk_timings)):
        actual_start, actual_end, theoretical_start, theoretical_end = timing
        logger.info(f"Chunk {i}: theoretical {theoretical_start}-{theoretical_end}s, actual {actual_start}-{actual_end}s")
        logger.info(f"Content preview: {content[:100]}...")

    # Test merging
    try:
        result_srt = merger.merge_chunk_results(chunk_results, chunk_timings)

        logger.info("\n=== Merged SRT Result ===")
        logger.info(result_srt)

        # Analyze results
        lines = result_srt.strip().split('\n')
        segment_count = len([line for line in lines if line.strip().isdigit()])

        logger.info(f"\n=== Analysis ===")
        logger.info(f"Total segments merged: {segment_count}")

        # Check for timing issues
        time_pattern = r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})'
        matches = re.findall(time_pattern, result_srt)

        timing_issues = 0
        for start_str, end_str in matches:
            start_sec = srt_time_to_seconds(start_str)
            end_sec = srt_time_to_seconds(end_str)
            if end_sec <= start_sec:
                timing_issues += 1
                logger.warning(f"Timing issue found: {start_str} --> {end_str}")

        if timing_issues == 0:
            logger.info("✅ No timing issues detected!")
        else:
            logger.error(f"❌ Found {timing_issues} timing issues!")

        return result_srt

    except Exception as e:
        logger.error(f"Error during merging: {e}")
        import traceback
        traceback.print_exc()
        return None

def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    time_part, ms_part = time_str.split(',')
    hours, minutes, seconds = time_part.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms_part) / 1000

def main():
    """Main test function"""
    logger.info("Starting SRT merging optimization tests")

    # Test SRT merging logic
    test_srt_merging()

    logger.info("=== Test completed ===")

if __name__ == "__main__":
    main()