// --- Enums ---

export type MediaType = 'image' | 'video';
export type SessionStatus = 'pending' | 'processing' | 'completed' | 'failed';
export type PostureClass =
  | 'Listening'
  | 'Looking Screen'
  | 'Hand Raising'
  | 'Sleeping / Head Down'
  | 'Turning Back'
  | 'Standing'
  | 'Writing'
  | 'Reading';

export const POSTURE_CLASSES: PostureClass[] = [
  'Listening',
  'Looking Screen',
  'Hand Raising',
  'Sleeping / Head Down',
  'Turning Back',
  'Standing',
  'Writing',
  'Reading',
];

export const POSTURE_COLORS: Record<PostureClass, string> = {
  Listening: '#22c55e',
  'Looking Screen': '#3b82f6',
  'Hand Raising': '#f59e0b',
  'Sleeping / Head Down': '#ef4444',
  'Turning Back': '#a855f7',
  Standing: '#ec4899',
  Writing: '#06b6d4',
  Reading: '#84cc16',
};

// --- API Response Types ---

export interface SessionData {
  id: string;
  filename: string;
  media_type: MediaType;
  status: SessionStatus;
  total_frames: number;
  processed_frames: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface StudentData {
  id: number;
  tracker_id: number;
  first_seen_frame: number;
  last_seen_frame: number;
  total_posture_changes: number;
}

export interface PostureRecordData {
  id: number;
  student_id: number;
  frame_id: number;
  posture: PostureClass;
  confidence: number;
  bbox_x1: number;
  bbox_y1: number;
  bbox_x2: number;
  bbox_y2: number;
  timestamp: string;
}

export interface PostureDistribution {
  posture: string;
  count: number;
  percentage: number;
}

export interface StudentSummary {
  tracker_id: number;
  current_posture: string | null;
  current_confidence: number;
  total_posture_changes: number;
  first_seen_frame: number;
  last_seen_frame: number;
}

export interface PostureSummaryData {
  session_id: string;
  total_students: number;
  total_frames: number;
  posture_distribution: PostureDistribution[];
  most_common_posture: string | null;
  average_confidence: number;
  students_summary: StudentSummary[];
}

export interface UploadResponse {
  session_id: string;
  message: string;
  status: SessionStatus;
}

export interface ProcessingProgress {
  session_id: string;
  status: SessionStatus;
  total_frames: number;
  processed_frames: number;
  progress_percent: number;
}

// --- Model metadata ---

export interface ModelDetails {
  model_path?: string;
  model_name?: string;
  task?: string;
  num_classes?: number;
  class_names?: Record<string, string>;
  total_parameters_millions?: number;
  input_size?: number;
  file_size_mb?: number;
}

export interface ModelInfoResponse {
  mode: 'inference' | 'demo';
  default_model?: string | null;
  available_models?: string[];
  info?: Record<string, ModelDetails>;
  message?: string;
  posture_classes?: string[];
}

// --- Detection overlay types ---

export interface DetectionOverlay {
  trackerId: number;
  bbox: [number, number, number, number];
  posture: PostureClass;
  confidence: number;
}
