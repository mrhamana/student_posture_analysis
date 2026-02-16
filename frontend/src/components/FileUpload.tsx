import React, { useCallback, useState, useRef } from 'react';
import { Upload, FileVideo, Image, AlertCircle } from 'lucide-react';
import clsx from 'clsx';

interface FileUploadProps {
  onUpload: (file: File) => Promise<string | null>;
  uploading: boolean;
  uploadProgress: number;
  uploadError: string | null;
}

const ACCEPTED_TYPES = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'video/mp4',
  'video/avi',
  'video/x-matroska',
  'video/quicktime',
];

const MAX_SIZE_MB = 500;

export const FileUpload: React.FC<FileUploadProps> = ({
  onUpload,
  uploading,
  uploadProgress,
  uploadError,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): string | null => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return `Unsupported file type: ${file.type || 'unknown'}. Accepted: JPEG, PNG, WebP, MP4, AVI, MKV, MOV`;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum: ${MAX_SIZE_MB}MB`;
    }
    return null;
  }, []);

  const handleFile = useCallback(
    async (file: File) => {
      setValidationError(null);
      const error = validateFile(file);
      if (error) {
        setValidationError(error);
        return;
      }
      await onUpload(file);
    },
    [onUpload, validateFile],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const error = validationError || uploadError;

  return (
    <div className="w-full">
      <div
        className={clsx(
          'relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition-all duration-200',
          dragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 bg-white hover:border-primary-400 hover:bg-gray-50',
          uploading && 'pointer-events-none opacity-60',
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <div className="mb-4 flex items-center gap-3">
          <div className="rounded-full bg-primary-100 p-3">
            <Upload className="h-6 w-6 text-primary-600" />
          </div>
          <div className="flex gap-2">
            <Image className="h-5 w-5 text-gray-400" />
            <FileVideo className="h-5 w-5 text-gray-400" />
          </div>
        </div>

        <p className="mb-1 text-base font-medium text-gray-700">
          {dragActive ? 'Drop your file here' : 'Drag & drop an image or video'}
        </p>
        <p className="mb-4 text-sm text-gray-500">or click to browse</p>

        <button
          type="button"
          className="btn-primary"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
        >
          Select File
        </button>

        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={ACCEPTED_TYPES.join(',')}
          onChange={handleChange}
          disabled={uploading}
        />

        <p className="mt-4 text-xs text-gray-400">
          Supported: JPEG, PNG, WebP, MP4, AVI, MKV, MOV — Max {MAX_SIZE_MB}MB
        </p>
      </div>

      {/* Upload Progress */}
      {uploading && (
        <div className="mt-4">
          <div className="mb-1 flex items-center justify-between text-sm">
            <span className="text-gray-600">Uploading...</span>
            <span className="font-medium text-primary-600">{uploadProgress}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
            <div
              className="h-full rounded-full bg-primary-500 transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-4 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
