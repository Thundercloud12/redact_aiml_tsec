// API service for AgriGuard backend communication
const API_BASE_URL = 'http://localhost:8000';

export interface ImageValidationResult {
  hash: string;
  not_empty: boolean;
  format_valid: boolean;
  format_info?: string;
  not_corrupted: boolean;
  visual_corruption: boolean;
  errors: string[];
  warnings: string[];
}

export interface PlantDiseaseResult {
  disease_detected: boolean;
  disease_type?: string;
  confidence_score: number;
  severity: 'mild' | 'moderate' | 'severe';
  treatment_recommendations: string[];
  grad_cam_visualization?: string;
}

class AgriGuardAPI {
  // Validate uploaded image for corruption and security
  async validateImage(file: File): Promise<ImageValidationResult> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/validate-image`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error validating image:', error);
      throw new Error('Failed to validate image. Please try again.');
    }
  }

  // Analyze plant disease (placeholder for future AI model integration)
  async analyzePlantDisease(file: File): Promise<PlantDiseaseResult> {
    // First validate the image
    const validation = await this.validateImage(file);
    
    if (!validation.not_corrupted) {
      throw new Error(`Image validation failed: ${validation.errors.join(', ')}`);
    }

    // For now, return mock data since the AI model integration would be added later
    // This structure is ready for when you integrate the actual plant disease detection model
    const mockResult: PlantDiseaseResult = {
      disease_detected: Math.random() > 0.3, // 70% chance of disease detection for demo
      disease_type: this.getRandomDisease(),
      confidence_score: Math.random() * 0.3 + 0.7, // 70-100% confidence
      severity: this.getRandomSeverity(),
      treatment_recommendations: this.getRandomTreatments(),
    };

    return mockResult;
  }

  // Helper methods for mock data
  private getRandomDisease(): string {
    const diseases = [
      'Early Blight',
      'Late Blight',
      'Leaf Spot',
      'Powdery Mildew',
      'Rust Disease',
      'Bacterial Wilt',
      'Mosaic Virus',
      'Root Rot'
    ];
    return diseases[Math.floor(Math.random() * diseases.length)];
  }

  private getRandomSeverity(): 'mild' | 'moderate' | 'severe' {
    const severities: ('mild' | 'moderate' | 'severe')[] = ['mild', 'moderate', 'severe'];
    return severities[Math.floor(Math.random() * severities.length)];
  }

  private getRandomTreatments(): string[] {
    const treatments = [
      'Apply copper-based fungicide spray',
      'Improve air circulation around plants',
      'Reduce watering frequency and water at soil level',
      'Remove infected plant parts immediately',
      'Apply organic neem oil treatment',
      'Increase spacing between plants',
      'Apply compost to improve soil health',
      'Use resistant plant varieties in future plantings'
    ];
    
    // Return 2-4 random treatments
    const numTreatments = Math.floor(Math.random() * 3) + 2;
    const shuffled = treatments.sort(() => 0.5 - Math.random());
    return shuffled.slice(0, numTreatments);
  }

  // Check if backend is available
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        timeout: 5000,
      } as RequestInit);
      return response.ok;
    } catch (error) {
      console.warn('Backend health check failed:', error);
      return false;
    }
  }
}

// Export singleton instance
export const agriGuardAPI = new AgriGuardAPI();

// Helper function to format file size
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Helper function to get file type
export const getFileType = (file: File): string => {
  return file.type.split('/')[1]?.toUpperCase() || 'Unknown';
};