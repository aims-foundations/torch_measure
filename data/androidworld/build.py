"""
Build AndroidWorld response matrices from publicly available per-task evaluation data.

AndroidWorld (Google Research, ICLR 2025) evaluates AI agents on 116 programmatic
tasks across 20 real-world Android apps. Tests mobile automation capabilities.

Data sources:
  - Official leaderboard: Google Sheets (community-submitted, self-reported)
    https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo
  - Per-task results scraped from agent benchmark pages:
    * DroidRun (91.4%): https://www.droidrun.ai/benchmark/
    * FinalRun (76.7%): https://www.finalrun.app/benchmark/
    * AutoDevice (94.8%): https://autodevice.io/benchmark/
  - Task list from paper appendix F (arXiv:2405.14573v4, ICLR 2025)
  - Trajectory file names from gbox.ai GitHub repo for canonical task names

Outputs:
  - response_matrix.csv: Binary (agents x tasks) matrix for agents with per-task data
  - leaderboard_summary.csv: All leaderboard agents with aggregate scores
  - task_metadata.csv: Task-level metadata (app, type, validation method, max steps)

Notes:
  - Only 3 agents have publicly available per-task pass/fail data
  - The official leaderboard has ~42 agents but only aggregate success rates
  - Per-task results are self-reported; no independent verification
  - AndroidWorld tasks are dynamically parameterized, so results may vary across runs
"""

INFO = {
    'description': """Build AndroidWorld response matrices from publicly available per-task evaluation data""",
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2405.14573',
    'data_source_url': """https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo""",
    'subject_type': 'agent',
    'item_type': 'task',
    'license': 'Apache-2.0',
    'citation': """@misc{rawles2025androidworlddynamicbenchmarkingenvironment,
      title={AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents}, 
      author={Christopher Rawles and Sarah Clinckemaillie and Yifan Chang and Jonathan Waltz and Gabrielle Lau and Marybeth Fair and Alice Li and William Bishop and Wei Li and Folawiyo Campbell-Ajala and Daniel Toyama and Robert Berry and Divya Tyamagundlu and Timothy Lillicrap and Oriana Riva},
      year={2025},
      eprint={2405.14573},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.14573}, 
}""",
    'tags': ['agent'],
}


import sys
from pathlib import Path
import os
import csv
import json
import subprocess
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================================
# Canonical list of 116 AndroidWorld tasks (from gbox.ai trajectory filenames
# and verified against paper Appendix F)
# ============================================================================

CANONICAL_TASKS = sorted([
    "AudioRecorderRecordAudio",
    "AudioRecorderRecordAudioWithFileName",
    "BrowserDraw",
    "BrowserMaze",
    "BrowserMultiply",
    "CameraTakePhoto",
    "CameraTakeVideo",
    "ClockStopWatchPausedVerify",
    "ClockStopWatchRunning",
    "ClockTimerEntry",
    "ContactsAddContact",
    "ContactsNewContactDraft",
    "ExpenseAddMultiple",
    "ExpenseAddMultipleFromGallery",
    "ExpenseAddMultipleFromMarkor",
    "ExpenseAddSingle",
    "ExpenseDeleteDuplicates",
    "ExpenseDeleteDuplicates2",
    "ExpenseDeleteMultiple",
    "ExpenseDeleteMultiple2",
    "ExpenseDeleteSingle",
    "FilesDeleteFile",
    "FilesMoveFile",
    "MarkorAddNoteHeader",
    "MarkorChangeNoteContent",
    "MarkorCreateFolder",
    "MarkorCreateNote",
    "MarkorCreateNoteAndSms",
    "MarkorCreateNoteFromClipboard",
    "MarkorDeleteAllNotes",
    "MarkorDeleteNewestNote",
    "MarkorDeleteNote",
    "MarkorEditNote",
    "MarkorMergeNotes",
    "MarkorMoveNote",
    "MarkorTranscribeReceipt",
    "MarkorTranscribeVideo",
    "NotesIsTodo",
    "NotesMeetingAttendeeCount",
    "NotesRecipeIngredientCount",
    "NotesTodoItemCount",
    "OpenAppTaskEval",
    "OsmAndFavorite",
    "OsmAndMarker",
    "OsmAndTrack",
    "RecipeAddMultipleRecipes",
    "RecipeAddMultipleRecipesFromImage",
    "RecipeAddMultipleRecipesFromMarkor",
    "RecipeAddMultipleRecipesFromMarkor2",
    "RecipeAddSingleRecipe",
    "RecipeDeleteDuplicateRecipes",
    "RecipeDeleteDuplicateRecipes2",
    "RecipeDeleteDuplicateRecipes3",
    "RecipeDeleteMultipleRecipes",
    "RecipeDeleteMultipleRecipesWithConstraint",
    "RecipeDeleteMultipleRecipesWithNoise",
    "RecipeDeleteSingleRecipe",
    "RecipeDeleteSingleWithRecipeWithNoise",
    "RetroCreatePlaylist",
    "RetroPlayingQueue",
    "RetroPlaylistDuration",
    "RetroSavePlaylist",
    "SaveCopyOfReceiptTaskEval",
    "SimpleCalendarAddOneEvent",
    "SimpleCalendarAddOneEventInTwoWeeks",
    "SimpleCalendarAddOneEventRelativeDay",
    "SimpleCalendarAddOneEventTomorrow",
    "SimpleCalendarAddRepeatingEvent",
    "SimpleCalendarAnyEventsOnDate",
    "SimpleCalendarDeleteEvents",
    "SimpleCalendarDeleteEventsOnRelativeDay",
    "SimpleCalendarDeleteOneEvent",
    "SimpleCalendarEventOnDateAtTime",
    "SimpleCalendarEventsInNextWeek",
    "SimpleCalendarEventsInTimeRange",
    "SimpleCalendarEventsOnDate",
    "SimpleCalendarFirstEventAfterStartTime",
    "SimpleCalendarLocationOfEvent",
    "SimpleCalendarNextEvent",
    "SimpleCalendarNextMeetingWithPerson",
    "SimpleDrawProCreateDrawing",
    "SimpleSmsReply",
    "SimpleSmsReplyMostRecent",
    "SimpleSmsResend",
    "SimpleSmsSend",
    "SimpleSmsSendClipboardContent",
    "SimpleSmsSendReceivedAddress",
    "SportsTrackerActivitiesCountForWeek",
    "SportsTrackerActivitiesOnDate",
    "SportsTrackerActivityDuration",
    "SportsTrackerLongestDistanceActivity",
    "SportsTrackerTotalDistanceForCategoryOverInterval",
    "SportsTrackerTotalDurationForCategoryThisWeek",
    "SystemBluetoothTurnOff",
    "SystemBluetoothTurnOffVerify",
    "SystemBluetoothTurnOn",
    "SystemBluetoothTurnOnVerify",
    "SystemBrightnessMax",
    "SystemBrightnessMaxVerify",
    "SystemBrightnessMin",
    "SystemBrightnessMinVerify",
    "SystemCopyToClipboard",
    "SystemWifiTurnOff",
    "SystemWifiTurnOffVerify",
    "SystemWifiTurnOn",
    "SystemWifiTurnOnVerify",
    "TasksCompletedTasksForDate",
    "TasksDueNextWeek",
    "TasksDueOnDate",
    "TasksHighPriorityTasks",
    "TasksHighPriorityTasksDueOnDate",
    "TasksIncompleteTasksOnDate",
    "TurnOffWifiAndTurnOnBluetooth",
    "TurnOnWifiAndOpenApp",
    "VlcCreatePlaylist",
    "VlcCreateTwoPlaylists",
])

assert len(CANONICAL_TASKS) == 116, f"Expected 116 tasks, got {len(CANONICAL_TASKS)}"


# ============================================================================
# Task metadata from paper Appendix F
# ============================================================================

TASK_METADATA = {
    "AudioRecorderRecordAudio": {"app": "audio_recorder", "type": "TC", "validation": "Filesystem", "max_steps": 12},
    "AudioRecorderRecordAudioWithFileName": {"app": "audio_recorder", "type": "TC", "validation": "Filesystem", "max_steps": 20},
    "BrowserDraw": {"app": "files,chrome", "type": "TC", "validation": "UI-elements", "max_steps": 20},
    "BrowserMaze": {"app": "files,chrome", "type": "TC", "validation": "UI-elements", "max_steps": 20},
    "BrowserMultiply": {"app": "files,chrome", "type": "TC", "validation": "UI-elements", "max_steps": 22},
    "CameraTakePhoto": {"app": "camera", "type": "TC", "validation": "Filesystem", "max_steps": 10},
    "CameraTakeVideo": {"app": "camera", "type": "TC", "validation": "Filesystem", "max_steps": 10},
    "ClockStopWatchPausedVerify": {"app": "clock", "type": "TC", "validation": "UI-elements", "max_steps": 10},
    "ClockStopWatchRunning": {"app": "clock", "type": "TC", "validation": "UI-elements", "max_steps": 10},
    "ClockTimerEntry": {"app": "clock", "type": "TC", "validation": "UI-elements", "max_steps": 10},
    "ContactsAddContact": {"app": "contacts", "type": "TC", "validation": "Database query", "max_steps": 12},
    "ContactsNewContactDraft": {"app": "contacts", "type": "TC", "validation": "UI-elements", "max_steps": 12},
    "ExpenseAddMultiple": {"app": "expense", "type": "TC", "validation": "Database query", "max_steps": 40},
    "ExpenseAddMultipleFromGallery": {"app": "gallery,expense", "type": "TC", "validation": "Database query", "max_steps": 20},
    "ExpenseAddMultipleFromMarkor": {"app": "markor,expense", "type": "TC", "validation": "Database query", "max_steps": 30},
    "ExpenseAddSingle": {"app": "expense", "type": "TC", "validation": "Database query", "max_steps": 12},
    "ExpenseDeleteDuplicates": {"app": "expense", "type": "TC", "validation": "Database query", "max_steps": 12},
    "ExpenseDeleteDuplicates2": {"app": "expense", "type": "TC", "validation": "Database query", "max_steps": 18},
    "ExpenseDeleteMultiple": {"app": "expense", "type": "TC", "validation": "Database query", "max_steps": 20},
    "ExpenseDeleteMultiple2": {"app": "expense", "type": "TC", "validation": "Database query", "max_steps": 34},
    "ExpenseDeleteSingle": {"app": "expense", "type": "TC", "validation": "Database query", "max_steps": 10},
    "FilesDeleteFile": {"app": "files", "type": "TC", "validation": "Filesystem", "max_steps": 10},
    "FilesMoveFile": {"app": "files", "type": "TC", "validation": "Filesystem", "max_steps": 20},
    "MarkorAddNoteHeader": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 12},
    "MarkorChangeNoteContent": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 12},
    "MarkorCreateFolder": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 10},
    "MarkorCreateNote": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 16},
    "MarkorCreateNoteAndSms": {"app": "markor,sms", "type": "TC", "validation": "Filesystem,Database query", "max_steps": 18},
    "MarkorCreateNoteFromClipboard": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 14},
    "MarkorDeleteAllNotes": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 14},
    "MarkorDeleteNewestNote": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 10},
    "MarkorDeleteNote": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 10},
    "MarkorEditNote": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 12},
    "MarkorMergeNotes": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 78},
    "MarkorMoveNote": {"app": "markor", "type": "TC", "validation": "Filesystem", "max_steps": 14},
    "MarkorTranscribeReceipt": {"app": "gallery,markor", "type": "TC", "validation": "Filesystem", "max_steps": 18},
    "MarkorTranscribeVideo": {"app": "markor,vlc", "type": "TC", "validation": "Filesystem", "max_steps": 20},
    "NotesIsTodo": {"app": "joplin", "type": "IR", "validation": "String match", "max_steps": 10},
    "NotesMeetingAttendeeCount": {"app": "joplin", "type": "IR", "validation": "String match", "max_steps": 10},
    "NotesRecipeIngredientCount": {"app": "joplin", "type": "IR", "validation": "String match", "max_steps": 10},
    "NotesTodoItemCount": {"app": "joplin", "type": "IR", "validation": "String match", "max_steps": 10},
    "OpenAppTaskEval": {"app": "camera,clock,contacts,settings,dialer", "type": "TC", "validation": "System API", "max_steps": 10},
    "OsmAndFavorite": {"app": "osmand", "type": "TC", "validation": "Filesystem", "max_steps": 13},
    "OsmAndMarker": {"app": "osmand", "type": "TC", "validation": "Filesystem", "max_steps": 20},
    "OsmAndTrack": {"app": "osmand", "type": "TC", "validation": "Filesystem", "max_steps": 120},
    "RecipeAddMultipleRecipes": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 68},
    "RecipeAddMultipleRecipesFromImage": {"app": "markor,recipe", "type": "TC", "validation": "Database query", "max_steps": 26},
    "RecipeAddMultipleRecipesFromMarkor": {"app": "gallery,recipe", "type": "TC", "validation": "Database query", "max_steps": 48},
    "RecipeAddMultipleRecipesFromMarkor2": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 52},
    "RecipeAddSingleRecipe": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 24},
    "RecipeDeleteDuplicateRecipes": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 10},
    "RecipeDeleteDuplicateRecipes2": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 24},
    "RecipeDeleteDuplicateRecipes3": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 34},
    "RecipeDeleteMultipleRecipes": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 24},
    "RecipeDeleteMultipleRecipesWithConstraint": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 40},
    "RecipeDeleteMultipleRecipesWithNoise": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 34},
    "RecipeDeleteSingleRecipe": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 10},
    "RecipeDeleteSingleWithRecipeWithNoise": {"app": "recipe", "type": "TC", "validation": "Database query", "max_steps": 20},
    "RetroCreatePlaylist": {"app": "music", "type": "TC", "validation": "Database query", "max_steps": 24},
    "RetroPlayingQueue": {"app": "music", "type": "TC", "validation": "Database query", "max_steps": 32},
    "RetroPlaylistDuration": {"app": "music", "type": "TC", "validation": "Database query", "max_steps": 30},
    "RetroSavePlaylist": {"app": "music", "type": "TC", "validation": "Database query", "max_steps": 50},
    "SaveCopyOfReceiptTaskEval": {"app": "gallery", "type": "TC", "validation": "Filesystem", "max_steps": 16},
    "SimpleCalendarAddOneEvent": {"app": "calendar", "type": "TC", "validation": "Database query", "max_steps": 34},
    "SimpleCalendarAddOneEventInTwoWeeks": {"app": "calendar", "type": "TC", "validation": "Database query", "max_steps": 20},
    "SimpleCalendarAddOneEventRelativeDay": {"app": "calendar", "type": "TC", "validation": "Database query", "max_steps": 34},
    "SimpleCalendarAddOneEventTomorrow": {"app": "calendar", "type": "TC", "validation": "Database query", "max_steps": 26},
    "SimpleCalendarAddRepeatingEvent": {"app": "calendar", "type": "TC", "validation": "Database query", "max_steps": 28},
    "SimpleCalendarAnyEventsOnDate": {"app": "calendar", "type": "IR", "validation": "Database query", "max_steps": 10},
    "SimpleCalendarDeleteEvents": {"app": "calendar", "type": "TC", "validation": "Database query", "max_steps": 14},
    "SimpleCalendarDeleteEventsOnRelativeDay": {"app": "calendar", "type": "TC", "validation": "Database query", "max_steps": 12},
    "SimpleCalendarDeleteOneEvent": {"app": "calendar", "type": "TC", "validation": "Database query", "max_steps": 12},
    "SimpleCalendarEventOnDateAtTime": {"app": "calendar", "type": "IR", "validation": "Database query", "max_steps": 10},
    "SimpleCalendarEventsInNextWeek": {"app": "calendar", "type": "IR", "validation": "Database query", "max_steps": 10},
    "SimpleCalendarEventsInTimeRange": {"app": "calendar", "type": "IR", "validation": "Database query", "max_steps": 10},
    "SimpleCalendarEventsOnDate": {"app": "calendar", "type": "IR", "validation": "Database query", "max_steps": 10},
    "SimpleCalendarFirstEventAfterStartTime": {"app": "calendar", "type": "IR", "validation": "Database query", "max_steps": 10},
    "SimpleCalendarLocationOfEvent": {"app": "calendar", "type": "IR", "validation": "Database query", "max_steps": 10},
    "SimpleCalendarNextEvent": {"app": "calendar", "type": "IR", "validation": "Database query", "max_steps": 10},
    "SimpleCalendarNextMeetingWithPerson": {"app": "calendar", "type": "IR", "validation": "Database query", "max_steps": 10},
    "SimpleDrawProCreateDrawing": {"app": "simpledrawpro", "type": "TC", "validation": "Filesystem", "max_steps": 18},
    "SimpleSmsReply": {"app": "sms", "type": "TC", "validation": "Database query", "max_steps": 12},
    "SimpleSmsReplyMostRecent": {"app": "sms", "type": "TC", "validation": "Database query", "max_steps": 12},
    "SimpleSmsResend": {"app": "sms", "type": "TC", "validation": "Database query", "max_steps": 12},
    "SimpleSmsSend": {"app": "sms", "type": "TC", "validation": "Database query", "max_steps": 12},
    "SimpleSmsSendClipboardContent": {"app": "sms", "type": "TC", "validation": "Database query", "max_steps": 12},
    "SimpleSmsSendReceivedAddress": {"app": "sms", "type": "TC", "validation": "Database query", "max_steps": 18},
    "SportsTrackerActivitiesCountForWeek": {"app": "sportstracker", "type": "IR", "validation": "String match", "max_steps": 10},
    "SportsTrackerActivitiesOnDate": {"app": "sportstracker", "type": "IR", "validation": "String match", "max_steps": 20},
    "SportsTrackerActivityDuration": {"app": "sportstracker", "type": "IR", "validation": "String match", "max_steps": 12},
    "SportsTrackerLongestDistanceActivity": {"app": "sportstracker", "type": "IR", "validation": "String match", "max_steps": 10},
    "SportsTrackerTotalDistanceForCategoryOverInterval": {"app": "sportstracker", "type": "IR", "validation": "String match", "max_steps": 22},
    "SportsTrackerTotalDurationForCategoryThisWeek": {"app": "sportstracker", "type": "IR", "validation": "String match", "max_steps": 16},
    "SystemBluetoothTurnOff": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemBluetoothTurnOffVerify": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemBluetoothTurnOn": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemBluetoothTurnOnVerify": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemBrightnessMax": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemBrightnessMaxVerify": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemBrightnessMin": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemBrightnessMinVerify": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemCopyToClipboard": {"app": "n/a", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemWifiTurnOff": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemWifiTurnOffVerify": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemWifiTurnOn": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "SystemWifiTurnOnVerify": {"app": "settings", "type": "TC", "validation": "System API", "max_steps": 10},
    "TasksCompletedTasksForDate": {"app": "tasks", "type": "IR", "validation": "String match", "max_steps": 10},
    "TasksDueNextWeek": {"app": "tasks", "type": "IR", "validation": "String match", "max_steps": 12},
    "TasksDueOnDate": {"app": "tasks", "type": "IR", "validation": "String match", "max_steps": 10},
    "TasksHighPriorityTasks": {"app": "tasks", "type": "IR", "validation": "String match", "max_steps": 10},
    "TasksHighPriorityTasksDueOnDate": {"app": "tasks", "type": "IR", "validation": "String match", "max_steps": 10},
    "TasksIncompleteTasksOnDate": {"app": "tasks", "type": "IR", "validation": "String match", "max_steps": 10},
    "TurnOffWifiAndTurnOnBluetooth": {"app": "settings", "type": "TC", "validation": "String match", "max_steps": 20},
    "TurnOnWifiAndOpenApp": {"app": "settings", "type": "TC", "validation": "String match", "max_steps": 20},
    "VlcCreatePlaylist": {"app": "vlc", "type": "TC", "validation": "String match", "max_steps": 28},
    "VlcCreateTwoPlaylists": {"app": "vlc", "type": "TC", "validation": "String match", "max_steps": 48},
}


# ============================================================================
# Per-task results from publicly available benchmark pages (scraped March 2026)
# Format: agent_name -> set of FAILED task names
# ============================================================================

# DroidRun (91.4% = 106/116 passed, 10 failed)
# Source: https://www.droidrun.ai/benchmark/
# Models: GPT-5 + Gemini 2.5 Pro, Screenshot + A11y tree
DROIDRUN_FAILED = {
    "ContactsNewContactDraft",
    "MarkorTranscribeVideo",
    "OsmAndMarker",
    "RecipeAddMultipleRecipesFromImage",
    "RecipeAddMultipleRecipesFromMarkor2",
    "RecipeDeleteDuplicateRecipes3",
    "RetroPlaylistDuration",
    "TasksCompletedTasksForDate",
    "TasksIncompleteTasksOnDate",
    # Note: DroidRun page listed 106 passed + 10 failed = 116
    # but only 9 failures were explicitly named. The BrowserDraw
    # task was not present in the passed list either, suggesting
    # it may be the 10th failure. We include it based on cross-check.
    "BrowserDraw",
}

# FinalRun (76.7% = 89/116 passed, 27 failed)
# Source: https://www.finalrun.app/benchmark/
# Model: GPT-5, Screenshot + A11y tree
FINALRUN_FAILED = {
    "BrowserMultiply",
    "ExpenseAddMultipleFromGallery",
    "ExpenseAddMultipleFromMarkor",
    "ExpenseDeleteDuplicates",
    "ExpenseDeleteDuplicates2",
    "MarkorAddNoteHeader",
    "MarkorCreateNoteFromClipboard",
    "MarkorEditNote",
    "MarkorMergeNotes",
    "MarkorTranscribeVideo",
    "OsmAndMarker",
    "OsmAndTrack",
    "RecipeAddMultipleRecipesFromMarkor2",
    "RecipeDeleteDuplicateRecipes2",
    "RecipeDeleteDuplicateRecipes3",
    "RecipeDeleteMultipleRecipesWithConstraint",
    "RetroPlaylistDuration",
    "SimpleCalendarDeleteEventsOnRelativeDay",
    "SimpleCalendarEventsInNextWeek",
    "SimpleCalendarEventsInTimeRange",
    "SystemBrightnessMax",
    "SystemBrightnessMaxVerify",
    "TasksCompletedTasksForDate",
    "TasksDueNextWeek",
    "TasksHighPriorityTasks",
    "VlcCreatePlaylist",
    "VlcCreateTwoPlaylists",
}

# AutoDevice (94.8% = 110/116 passed, 6 failed)
# Source: https://autodevice.io/benchmark/
# Models: Gemini 3 Pro + Sonnet 4.5, Screenshot
AUTODEVICE_FAILED = {
    "MarkorAddNoteHeader",
    "MarkorChangeNoteContent",
    "MarkorMergeNotes",
    "MarkorTranscribeVideo",
    "RecipeDeleteDuplicateRecipes2",
    "RecipeDeleteDuplicateRecipes3",
}


# ============================================================================
# Agent metadata for per-task results
# ============================================================================

PERTASK_AGENTS = {
    "DroidRun": {
        "failed_tasks": DROIDRUN_FAILED,
        "release_date": "10/2025",
        "model": "GPT-5 + Gemini 2.5 Pro",
        "screen_repr": "Screenshot + A11y tree",
        "model_type": "AI agent",
        "open_source": True,
        "aggregate_score": 91.4,
    },
    "FinalRun": {
        "failed_tasks": FINALRUN_FAILED,
        "release_date": "08/2025",
        "model": "GPT-5",
        "screen_repr": "Screenshot + A11y tree",
        "model_type": "AI agent",
        "open_source": False,
        "aggregate_score": 76.7,
    },
    "AutoDevice": {
        "failed_tasks": AUTODEVICE_FAILED,
        "release_date": "01/2026",
        "model": "Gemini 3 Pro + Sonnet 4.5",
        "screen_repr": "Screenshot",
        "model_type": "AI agent",
        "open_source": True,
        "aggregate_score": 94.8,
    },
}


def download_leaderboard():
    """Download the official AndroidWorld leaderboard from Google Sheets."""
    url = ("https://docs.google.com/spreadsheets/d/"
           "1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/"
           "export?format=csv&gid=0")
    output_path = os.path.join(RAW_DIR, "leaderboard.csv")

    print("Downloading official AndroidWorld leaderboard...")
    try:
        result = subprocess.run(
            ["curl", "-s", "-L", url, "-o", output_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"  Saved: {output_path} ({size / 1024:.1f} KB)")
            return output_path
    except Exception as e:
        print(f"  Download failed: {e}")

    if os.path.exists(output_path):
        print(f"  Using cached: {output_path}")
        return output_path

    print("  WARNING: Leaderboard not available")
    return None


def save_per_task_data():
    """Save per-task agent results to raw JSON files."""
    for agent_name, info in PERTASK_AGENTS.items():
        results = {}
        for task in CANONICAL_TASKS:
            results[task] = 0 if task in info["failed_tasks"] else 1

        # Verify counts
        n_pass = sum(results.values())
        n_fail = len(CANONICAL_TASKS) - n_pass
        expected_score = n_pass / len(CANONICAL_TASKS) * 100

        output_path = os.path.join(RAW_DIR, f"pertask_{agent_name.lower()}.json")
        data = {
            "agent": agent_name,
            "release_date": info["release_date"],
            "model": info["model"],
            "screen_repr": info["screen_repr"],
            "model_type": info["model_type"],
            "open_source": info["open_source"],
            "reported_score": info["aggregate_score"],
            "computed_score": round(expected_score, 1),
            "n_tasks": len(CANONICAL_TASKS),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "per_task_results": results,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {output_path}")
        print(f"    {agent_name}: {n_pass}/{len(CANONICAL_TASKS)} = "
              f"{expected_score:.1f}% (reported: {info['aggregate_score']}%)")


def build_response_matrix():
    """Build the response matrix from per-task data."""
    agent_names = sorted(PERTASK_AGENTS.keys())
    n_agents = len(agent_names)
    n_tasks = len(CANONICAL_TASKS)

    # Build matrix: agents x tasks
    matrix = np.zeros((n_agents, n_tasks), dtype=int)
    for i, agent in enumerate(agent_names):
        failed = PERTASK_AGENTS[agent]["failed_tasks"]
        for j, task in enumerate(CANONICAL_TASKS):
            matrix[i, j] = 0 if task in failed else 1

    # Create DataFrame
    matrix_df = pd.DataFrame(matrix, index=agent_names, columns=CANONICAL_TASKS)
    matrix_df.index.name = "Agent"

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)

    # Print statistics
    total_cells = n_agents * n_tasks
    n_pass = int(matrix.sum())
    n_fail = total_cells - n_pass
    mean_pass_rate = matrix.mean()

    print(f"\n{'=' * 60}")
    print(f"  RESPONSE MATRIX (Per-Task Data)")
    print(f"{'=' * 60}")
    print(f"  Agents:          {n_agents}")
    print(f"  Tasks:           {n_tasks}")
    print(f"  Matrix dims:     {n_agents} x {n_tasks}")
    print(f"  Total cells:     {total_cells:,}")
    print(f"  Pass cells:      {n_pass:,} ({n_pass / total_cells * 100:.1f}%)")
    print(f"  Fail cells:      {n_fail:,} ({n_fail / total_cells * 100:.1f}%)")
    print(f"  Fill rate:       100.0%")
    print(f"  Mean pass rate:  {mean_pass_rate * 100:.1f}%")

    # Per-agent stats
    per_agent_pass = matrix.mean(axis=1)
    print(f"\n  Per-agent pass rate:")
    print(f"    Min:    {per_agent_pass.min() * 100:.1f}% "
          f"({agent_names[per_agent_pass.argmin()]})")
    print(f"    Max:    {per_agent_pass.max() * 100:.1f}% "
          f"({agent_names[per_agent_pass.argmax()]})")
    print(f"    Median: {np.median(per_agent_pass) * 100:.1f}%")
    print(f"    Std:    {per_agent_pass.std() * 100:.1f}%")

    for i, agent in enumerate(agent_names):
        score = per_agent_pass[i] * 100
        reported = PERTASK_AGENTS[agent]["aggregate_score"]
        model = PERTASK_AGENTS[agent]["model"]
        print(f"    {agent:20s}  {score:.1f}% (reported: {reported}%)  "
              f"[{model}]")

    # Per-task stats
    per_task_solve = matrix.mean(axis=0)
    print(f"\n  Per-task solve rate (across {n_agents} agents):")
    print(f"    Min:    {per_task_solve.min() * 100:.1f}%")
    print(f"    Max:    {per_task_solve.max() * 100:.1f}%")
    print(f"    Median: {np.median(per_task_solve) * 100:.1f}%")
    print(f"    Std:    {per_task_solve.std() * 100:.1f}%")

    # Task difficulty distribution
    solved_by_all = (per_task_solve == 1.0).sum()
    solved_by_none = (per_task_solve == 0.0).sum()
    print(f"\n  Task difficulty distribution:")
    print(f"    Solved by ALL agents:    {solved_by_all}")
    print(f"    Solved by NO agents:     {solved_by_none}")
    for k in range(n_agents + 1):
        count = (per_task_solve == k / n_agents).sum()
        print(f"    Solved by {k}/{n_agents} agents:      {count}")

    # Hardest tasks (failed by most agents)
    task_fail_count = n_agents - matrix.sum(axis=0)
    hard_idx = np.argsort(-task_fail_count)
    print(f"\n  Hardest tasks (most failures):")
    for idx in hard_idx[:15]:
        task = CANONICAL_TASKS[idx]
        fails = int(task_fail_count[idx])
        if fails == 0:
            break
        failed_by = [agent_names[i] for i in range(n_agents)
                     if matrix[i, idx] == 0]
        print(f"    {task:50s}  failed by {fails}/{n_agents}: "
              f"{', '.join(failed_by)}")

    print(f"\n  Saved: {output_path}")
    return matrix_df


def build_task_metadata():
    """Build and save task metadata CSV."""
    rows = []
    for task in CANONICAL_TASKS:
        meta = TASK_METADATA.get(task, {})
        # Determine primary app
        apps = meta.get("app", "unknown")
        primary_app = apps.split(",")[0] if apps else "unknown"

        rows.append({
            "task_id": task,
            "primary_app": primary_app,
            "all_apps": apps,
            "task_type": meta.get("type", ""),
            "validation_method": meta.get("validation", ""),
            "max_steps": meta.get("max_steps", ""),
        })

    metadata_df = pd.DataFrame(rows)
    output_path = os.path.join(PROCESSED_DIR, "task_metadata.csv")
    metadata_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"  TASK METADATA")
    print(f"{'=' * 60}")
    print(f"  Total tasks: {len(metadata_df)}")

    # App distribution
    print(f"\n  Tasks per primary app:")
    app_counts = metadata_df["primary_app"].value_counts()
    for app, count in app_counts.items():
        print(f"    {app:25s}  {count}")

    # Task type distribution
    print(f"\n  Task type distribution:")
    type_counts = metadata_df["task_type"].value_counts()
    for ttype, count in type_counts.items():
        label = "Task Completion" if ttype == "TC" else "Information Retrieval"
        print(f"    {ttype} ({label}):  {count}")

    # Validation method distribution
    print(f"\n  Validation method distribution:")
    val_counts = metadata_df["validation_method"].value_counts()
    for val, count in val_counts.items():
        print(f"    {val:25s}  {count}")

    # Max steps distribution
    steps = metadata_df["max_steps"].astype(int)
    print(f"\n  Max steps distribution:")
    print(f"    Min:    {steps.min()}")
    print(f"    Max:    {steps.max()}")
    print(f"    Median: {steps.median():.0f}")
    print(f"    Mean:   {steps.mean():.1f}")

    print(f"\n  Saved: {output_path}")
    return metadata_df


def build_leaderboard_summary(leaderboard_path):
    """Parse the official leaderboard CSV and build a summary."""
    if not leaderboard_path or not os.path.exists(leaderboard_path):
        print("\n  Skipping leaderboard summary (no data)")
        return None

    # Read the CSV, skipping the warning header row
    with open(leaderboard_path, "r") as f:
        lines = f.readlines()

    # Find the header row (contains "Rank")
    header_idx = None
    for i, line in enumerate(lines):
        if "Rank" in line and "Model" in line:
            header_idx = i
            break

    if header_idx is None:
        print("  WARNING: Could not find header row in leaderboard CSV")
        return None

    # Parse from header onward
    rows = []
    reader = csv.reader(lines[header_idx:])
    header = next(reader)

    # Clean header names
    header = [h.strip().replace("\n", " ") for h in header]

    for row in reader:
        if not row or not row[0].strip():
            continue
        # Skip non-data rows
        rank = row[0].strip()
        if not rank.isdigit():
            # Check for "Human Performance" or definition rows
            if "Human" in str(row):
                pass  # Include human baseline
            else:
                continue

        if len(row) < 9:
            continue

        try:
            success_rate = float(row[8].strip()) if row[8].strip() else None
        except (ValueError, IndexError):
            success_rate = None

        if success_rate is None:
            continue

        rows.append({
            "rank": rank,
            "release_date": row[1].strip() if len(row) > 1 else "",
            "source": row[2].strip() if len(row) > 2 else "",
            "model_type": row[3].strip() if len(row) > 3 else "",
            "open_source": row[4].strip() if len(row) > 4 else "",
            "model_size": row[5].strip() if len(row) > 5 else "",
            "model": row[6].strip() if len(row) > 6 else "",
            "screen_repr": row[7].strip() if len(row) > 7 else "",
            "success_rate": success_rate,
            "num_trials": row[9].strip() if len(row) > 9 else "",
            "pass_at_k": row[10].strip() if len(row) > 10 else "",
        })

    if not rows:
        print("  WARNING: No valid entries found in leaderboard")
        return None

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("success_rate", ascending=False)

    output_path = os.path.join(PROCESSED_DIR, "leaderboard_summary.csv")
    summary_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"  LEADERBOARD SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total entries: {len(summary_df)}")
    print(f"  Score range: {summary_df['success_rate'].min():.1f}% - "
          f"{summary_df['success_rate'].max():.1f}%")
    print(f"  Median score: {summary_df['success_rate'].median():.1f}%")
    print(f"  Mean score:   {summary_df['success_rate'].mean():.1f}%")

    # Model type distribution
    print(f"\n  Model type distribution:")
    for mt, count in summary_df["model_type"].value_counts().items():
        print(f"    {mt:20s}  {count}")

    # Open source distribution
    print(f"\n  Open source:")
    for os_flag, count in summary_df["open_source"].value_counts().items():
        label = "Yes" if os_flag == "\u2714" else "No" if os_flag == "\u2717" else os_flag
        print(f"    {label:20s}  {count}")

    # Top 10
    print(f"\n  Top 10 entries:")
    for _, r in summary_df.head(10).iterrows():
        print(f"    {r['source']:35s}  {r['success_rate']:5.1f}%  "
              f"[{r['model']}]")

    # Entries with per-task data available
    print(f"\n  Entries with per-task data available in this dataset:")
    pertask_names = set(PERTASK_AGENTS.keys())
    for _, r in summary_df.iterrows():
        source = r["source"]
        if any(name.lower() in source.lower() for name in pertask_names):
            print(f"    {source:35s}  {r['success_rate']:5.1f}%")

    print(f"\n  Saved: {output_path}")
    return summary_df


def main():
    print("AndroidWorld Response Matrix Builder")
    print("=" * 60)
    print()
    print("AndroidWorld: 116 tasks across 20 Android apps")
    print("Paper: https://arxiv.org/abs/2405.14573 (ICLR 2025)")
    print("Leaderboard: https://docs.google.com/spreadsheets/d/"
          "1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo")
    print()

    # Step 1: Download leaderboard
    leaderboard_path = download_leaderboard()

    # Step 2: Save per-task data to raw/
    print("\nSaving per-task results to raw/...")
    save_per_task_data()

    # Step 3: Build response matrix
    matrix_df = build_response_matrix()

    # Step 4: Build task metadata
    metadata_df = build_task_metadata()

    # Step 5: Build leaderboard summary
    leaderboard_df = build_leaderboard_summary(leaderboard_path)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  PRIMARY response matrix:")
    print(f"    Dimensions: {matrix_df.shape[0]} agents x "
          f"{matrix_df.shape[1]} tasks")
    print(f"    Fill rate:  100.0%")
    print(f"    Mean pass:  {matrix_df.values.mean() * 100:.1f}%")
    print(f"    Agents with per-task data: {matrix_df.shape[0]}")
    if leaderboard_df is not None:
        print(f"    Agents on leaderboard (aggregate only): "
              f"{len(leaderboard_df)}")
    print()
    print("  DATA AVAILABILITY NOTE:")
    print("  Only 3 out of ~42 leaderboard agents have publicly available")
    print("  per-task pass/fail data. The remaining agents report only")
    print("  aggregate success rates. The response matrix includes only")
    print("  agents with per-task data. The leaderboard_summary.csv")
    print("  contains all agents with their aggregate scores.")
    print()
    print("  To expand the response matrix, options include:")
    print("  1. Download trajectory .pkl files from agents that submitted")
    print("     them (gbox.ai, Surfer 2, K2-Agent) and extract pass/fail")
    print("  2. Contact agent authors for per-task results")
    print("  3. Run agents locally using the AndroidWorld environment")
    print()

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")

    for f in sorted(os.listdir(RAW_DIR)):
        fpath = os.path.join(RAW_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    raw/{f:41s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    import os, subprocess
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
