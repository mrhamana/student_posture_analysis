import { useReducer, useCallback, useRef, useEffect } from 'react';
import { api } from '../api/client';
import type {
  SessionData,
  PostureSummaryData,
  PostureRecordData,
  StudentData,
  UploadResponse,
  ProcessingProgress,
  SessionStatus,
} from '../types';

// --- State Shape ---

interface AppState {
  // Upload
  uploading: boolean;
  uploadProgress: number;
  uploadError: string | null;

  // Current session
  currentSessionId: string | null;
  session: SessionData | null;
  sessionLoading: boolean;
  sessionError: string | null;

  // Processing
  processingProgress: ProcessingProgress | null;

  // Results
  students: StudentData[];
  records: PostureRecordData[];
  summary: PostureSummaryData | null;
  resultsLoading: boolean;
  resultsError: string | null;

  // Session list
  sessions: SessionData[];
  sessionsTotal: number;
  sessionsLoading: boolean;
}

const initialState: AppState = {
  uploading: false,
  uploadProgress: 0,
  uploadError: null,
  currentSessionId: null,
  session: null,
  sessionLoading: false,
  sessionError: null,
  processingProgress: null,
  students: [],
  records: [],
  summary: null,
  resultsLoading: false,
  resultsError: null,
  sessions: [],
  sessionsTotal: 0,
  sessionsLoading: false,
};

// --- Actions ---

type Action =
  | { type: 'UPLOAD_START' }
  | { type: 'UPLOAD_PROGRESS'; payload: number }
  | { type: 'UPLOAD_SUCCESS'; payload: UploadResponse }
  | { type: 'UPLOAD_ERROR'; payload: string }
  | { type: 'SET_SESSION_LOADING' }
  | { type: 'SET_SESSION'; payload: SessionData }
  | { type: 'SET_SESSION_ERROR'; payload: string }
  | { type: 'SET_PROGRESS'; payload: ProcessingProgress }
  | { type: 'SET_RESULTS_LOADING' }
  | { type: 'SET_RESULTS'; payload: { students: StudentData[]; records: PostureRecordData[]; summary: PostureSummaryData } }
  | { type: 'SET_RESULTS_ERROR'; payload: string }
  | { type: 'SET_SESSIONS_LOADING' }
  | { type: 'SET_SESSIONS'; payload: { sessions: SessionData[]; total: number } }
  | { type: 'SELECT_SESSION'; payload: string }
  | { type: 'RESET' };

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'UPLOAD_START':
      return { ...state, uploading: true, uploadProgress: 0, uploadError: null };
    case 'UPLOAD_PROGRESS':
      return { ...state, uploadProgress: action.payload };
    case 'UPLOAD_SUCCESS':
      return {
        ...state,
        uploading: false,
        uploadProgress: 100,
        currentSessionId: action.payload.session_id,
      };
    case 'UPLOAD_ERROR':
      return { ...state, uploading: false, uploadError: action.payload };
    case 'SET_SESSION_LOADING':
      return { ...state, sessionLoading: true, sessionError: null };
    case 'SET_SESSION':
      return { ...state, session: action.payload, sessionLoading: false };
    case 'SET_SESSION_ERROR':
      return { ...state, sessionError: action.payload, sessionLoading: false };
    case 'SET_PROGRESS':
      return { ...state, processingProgress: action.payload };
    case 'SET_RESULTS_LOADING':
      return { ...state, resultsLoading: true, resultsError: null };
    case 'SET_RESULTS':
      return {
        ...state,
        students: action.payload.students,
        records: action.payload.records,
        summary: action.payload.summary,
        resultsLoading: false,
      };
    case 'SET_RESULTS_ERROR':
      return { ...state, resultsError: action.payload, resultsLoading: false };
    case 'SET_SESSIONS_LOADING':
      return { ...state, sessionsLoading: true };
    case 'SET_SESSIONS':
      return {
        ...state,
        sessions: action.payload.sessions,
        sessionsTotal: action.payload.total,
        sessionsLoading: false,
      };
    case 'SELECT_SESSION':
      return {
        ...state,
        currentSessionId: action.payload,
        session: null,
        students: [],
        records: [],
        summary: null,
        processingProgress: null,
      };
    case 'RESET':
      return initialState;
    default:
      return state;
  }
}

// --- Hook ---

export function useAppState() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const uploadFile = useCallback(async (file: File) => {
    dispatch({ type: 'UPLOAD_START' });
    try {
      const result = await api.uploadMedia(file, (percent) => {
        dispatch({ type: 'UPLOAD_PROGRESS', payload: percent });
      });
      dispatch({ type: 'UPLOAD_SUCCESS', payload: result });
      return result.session_id;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Upload failed';
      dispatch({ type: 'UPLOAD_ERROR', payload: message });
      return null;
    }
  }, []);

  const loadSession = useCallback(async (sessionId: string) => {
    dispatch({ type: 'SET_SESSION_LOADING' });
    try {
      const session = await api.getSession(sessionId);
      dispatch({ type: 'SET_SESSION', payload: session });
      return session;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load session';
      dispatch({ type: 'SET_SESSION_ERROR', payload: message });
      return null;
    }
  }, []);

  const loadResults = useCallback(async (sessionId: string) => {
    dispatch({ type: 'SET_RESULTS_LOADING' });
    try {
      const [studentsRes, recordsRes, summary] = await Promise.all([
        api.getSessionStudents(sessionId),
        api.getSessionRecords(sessionId),
        api.getPostureSummary(sessionId),
      ]);
      dispatch({
        type: 'SET_RESULTS',
        payload: {
          students: studentsRes.students,
          records: recordsRes.records,
          summary,
        },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load results';
      dispatch({ type: 'SET_RESULTS_ERROR', payload: message });
    }
  }, []);

  const pollProgress = useCallback(
    (sessionId: string) => {
      // Clear existing polling
      if (pollingRef.current) clearInterval(pollingRef.current);

      const poll = async () => {
        try {
          const progress = await api.getSessionProgress(sessionId);
          dispatch({ type: 'SET_PROGRESS', payload: progress });

          // Also refresh session data
          const session = await api.getSession(sessionId);
          dispatch({ type: 'SET_SESSION', payload: session });

          if (progress.status === 'completed' || progress.status === 'failed') {
            if (pollingRef.current) clearInterval(pollingRef.current);
            pollingRef.current = null;

            if (progress.status === 'completed') {
              await loadResults(sessionId);
            }
          }
        } catch {
          // continue polling on error
        }
      };

      // Initial poll
      poll();
      pollingRef.current = setInterval(poll, 2000);
    },
    [loadResults],
  );

  const selectSession = useCallback(
    async (sessionId: string) => {
      dispatch({ type: 'SELECT_SESSION', payload: sessionId });
      const session = await loadSession(sessionId);
      if (session) {
        if (session.status === 'completed') {
          await loadResults(sessionId);
        } else if (session.status === 'processing' || session.status === 'pending') {
          pollProgress(sessionId);
        }
      }
    },
    [loadSession, loadResults, pollProgress],
  );

  const loadSessions = useCallback(async () => {
    dispatch({ type: 'SET_SESSIONS_LOADING' });
    try {
      const data = await api.getSessions();
      dispatch({ type: 'SET_SESSIONS', payload: data });
    } catch {
      // silently fail sessions list
    }
  }, []);

  const reset = useCallback(() => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    dispatch({ type: 'RESET' });
  }, []);

  return {
    state,
    uploadFile,
    loadSession,
    loadResults,
    pollProgress,
    selectSession,
    loadSessions,
    reset,
  };
}
