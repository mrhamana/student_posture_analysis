export interface StudentProfile {
    trackerId: number;
    name: string;
    imageUrl: string;
}

const CUSTOM_PROFILES: Record<number, { name?: string; imageUrl?: string }> = {
    // Add overrides here if you have named students or custom image paths.
    // Example: 1: { name: 'Alice Sharma', imageUrl: '/students/id1.jpg' },
};

export const getStudentProfile = (trackerId: number): StudentProfile => {
    const custom = CUSTOM_PROFILES[trackerId];
    return {
        trackerId,
        name: custom?.name ?? `Student ${trackerId}`,
        imageUrl: custom?.imageUrl ?? `/students/id${trackerId}.jpg`,
    };
};
