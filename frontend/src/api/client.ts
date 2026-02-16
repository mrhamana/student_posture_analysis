import axios, { AxiosError, AxiosInstance } from 'axios';
import type {
  SessionData,
  StudentData,
  PostureSummaryData,
  UploadResponse,
  ProcessingProgress,
  PostureRecordData,
  ModelInfoResponse,
} from '../types';

const BASE_URL = '/api';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: BASE_URL,
      timeout: 60000,
      headers: { 'Content-Type': 'application/json' },
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const message =
          (error.response?.data as { detail?: string })?.detail ||
          error.message ||
          'An unexpected error occurred';
        console.error('[API Error]', message);
        return Promise.reject(new Error(message));
      },
    );
  }

  // --- Upload ---

  async uploadMedia(
    file: File,
    modelName?: string | null,
    onProgress?: (percent: number) => void,
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const params: Record<string, string> = {};
    if (modelName) {
      params.model = modelName;
    }

    const response = await this.client.post<UploadResponse>('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 min for large files
      params,
      onUploadProgress: (event) => {
        if (event.total && onProgress) {
          onProgress(Math.round((event.loaded / event.total) * 100));
        }
      },
    });
    return response.data;
  }

  // --- Models ---

  async getModelInfo(): Promise<ModelInfoResponse> {
    const response = await this.client.get<ModelInfoResponse>('/models');
    return response.data;
  }

  // --- Sessions ---

  async getSessions(skip = 0, limit = 50): Promise<{ sessions: SessionData[]; total: number }> {
    const response = await this.client.get('/sessions', { params: { skip, limit } });
    return response.data;
  }

  async getSession(sessionId: string): Promise<SessionData> {
    const response = await this.client.get<SessionData>(`/session/${sessionId}`);
    return response.data;
  }

  async getSessionProgress(sessionId: string): Promise<ProcessingProgress> {
    const response = await this.client.get<ProcessingProgress>(`/session/${sessionId}/progress`);
    return response.data;
  }

  async deleteSession(sessionId: string): Promise<void> {
    await this.client.delete(`/session/${sessionId}`);
  }

  // --- Students ---

  async getSessionStudents(sessionId: string): Promise<{ students: StudentData[]; total: number }> {
    const response = await this.client.get(`/session/${sessionId}/students`);
    return response.data;
  }

  // --- Posture ---

  async getPostureSummary(sessionId: string): Promise<PostureSummaryData> {
    const response = await this.client.get<PostureSummaryData>(
      `/session/${sessionId}/posture-summary`,
    );
    return response.data;
  }

  async getSessionRecords(
    sessionId: string,
  ): Promise<{ records: PostureRecordData[]; total: number }> {
    const response = await this.client.get(`/session/${sessionId}/records`);
    return response.data;
  }
}

export const api = new ApiClient();
