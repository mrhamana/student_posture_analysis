import React, { useMemo, useState, useCallback } from 'react';
import { Search, ArrowUpDown, Filter } from 'lucide-react';
import clsx from 'clsx';
import type { PostureSummaryData, PostureClass } from '../types';
import { POSTURE_CLASSES, POSTURE_COLORS } from '../types';

interface PostureTableProps {
  summary: PostureSummaryData;
}

type SortField = 'tracker_id' | 'current_posture' | 'current_confidence' | 'total_posture_changes';
type SortDir = 'asc' | 'desc';

export const PostureTable: React.FC<PostureTableProps> = ({ summary }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [postureFilter, setPostureFilter] = useState<string>('all');
  const [sortField, setSortField] = useState<SortField>('tracker_id');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const handleSort = useCallback(
    (field: SortField) => {
      if (sortField === field) {
        setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
      } else {
        setSortField(field);
        setSortDir('asc');
      }
    },
    [sortField],
  );

  const filteredStudents = useMemo(() => {
    let students = [...summary.students_summary];

    // Search filter
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      students = students.filter((s) => s.tracker_id.toString().includes(q));
    }

    // Posture filter
    if (postureFilter !== 'all') {
      students = students.filter((s) => s.current_posture === postureFilter);
    }

    // Sort
    students.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case 'tracker_id':
          cmp = a.tracker_id - b.tracker_id;
          break;
        case 'current_posture':
          cmp = (a.current_posture || '').localeCompare(b.current_posture || '');
          break;
        case 'current_confidence':
          cmp = a.current_confidence - b.current_confidence;
          break;
        case 'total_posture_changes':
          cmp = a.total_posture_changes - b.total_posture_changes;
          break;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return students;
  }, [summary.students_summary, searchQuery, postureFilter, sortField, sortDir]);

  const SortHeader: React.FC<{ field: SortField; children: React.ReactNode }> = ({
    field,
    children,
  }) => (
    <th
      className="cursor-pointer select-none px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-600 transition-colors hover:text-primary-600"
      onClick={() => handleSort(field)}
    >
      <div className="flex items-center gap-1">
        {children}
        <ArrowUpDown
          className={clsx(
            'h-3 w-3',
            sortField === field ? 'text-primary-500' : 'text-gray-400',
          )}
        />
      </div>
    </th>
  );

  return (
    <div className="card">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Student Posture Details</h3>

        <div className="flex flex-col gap-2 sm:flex-row">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search by ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full rounded-lg border border-gray-300 bg-white py-2 pl-9 pr-3 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500 sm:w-40"
            />
          </div>

          {/* Posture filter */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <select
              value={postureFilter}
              onChange={(e) => setPostureFilter(e.target.value)}
              className="w-full appearance-none rounded-lg border border-gray-300 bg-white py-2 pl-9 pr-8 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
            >
              <option value="all">All Postures</option>
              {POSTURE_CLASSES.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <SortHeader field="tracker_id">Student ID</SortHeader>
              <SortHeader field="current_posture">Current Posture</SortHeader>
              <SortHeader field="current_confidence">Confidence</SortHeader>
              <SortHeader field="total_posture_changes">Posture Changes</SortHeader>
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-600">
                Frame Range
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {filteredStudents.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-gray-500">
                  No students match the current filters
                </td>
              </tr>
            ) : (
              filteredStudents.map((student) => {
                const postureColor =
                  POSTURE_COLORS[student.current_posture as PostureClass] || '#6b7280';
                return (
                  <tr
                    key={student.tracker_id}
                    className="transition-colors hover:bg-gray-50"
                  >
                    <td className="px-4 py-3 font-medium text-gray-900">
                      #{student.tracker_id}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className="badge"
                        style={{
                          backgroundColor: `${postureColor}20`,
                          color: postureColor,
                        }}
                      >
                        {student.current_posture || 'N/A'}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 overflow-hidden rounded-full bg-gray-200">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${student.current_confidence * 100}%`,
                              backgroundColor: postureColor,
                            }}
                          />
                        </div>
                        <span className="text-xs text-gray-600">
                          {(student.current_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-gray-700">{student.total_posture_changes}</td>
                    <td className="px-4 py-3 text-xs text-gray-500">
                      {student.first_seen_frame} – {student.last_seen_frame}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <p className="mt-2 text-xs text-gray-400">
        Showing {filteredStudents.length} of {summary.students_summary.length} students
      </p>
    </div>
  );
};
