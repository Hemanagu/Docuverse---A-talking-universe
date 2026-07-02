'use client';
import * as React from 'react';
import { Upload, CheckCircle, XCircle, Loader2 } from 'lucide-react';

const FileUploadComponent: React.FC = () => {

  const [uploadStatus, setUploadStatus] = React.useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [uploadedFileName, setUploadedFileName] = React.useState<string>('');
  const [errorMessage, setErrorMessage] = React.useState<string>('');

  const handleFileUploadButtonClick = () => {

    const el = document.createElement('input');
    el.setAttribute('type', 'file');
    el.setAttribute('accept', 'application/pdf');

    el.addEventListener('change', async () => {

      if (el.files && el.files.length > 0) {
        const file = el.files.item(0);

        if (file) {
          // Validate file size (max 10MB)
          const maxSize = 10 * 1024 * 1024; // 10MB
          if (file.size > maxSize) {
            setUploadStatus('error');
            setErrorMessage('File size exceeds 10MB limit');
            return;
          }

          // Validate file type
          if (file.type !== 'application/pdf') {
            setUploadStatus('error');
            setErrorMessage('Please upload a PDF file');
            return;
          }

          try {

            setUploadStatus('uploading');
            setUploadedFileName(file.name);
            setErrorMessage('');

            // Upload file
            const formData = new FormData();
            formData.append('pdf', file);

            const response = await fetch('http://localhost:8000/upload/pdf', {
              method: 'POST',
              body: formData,
            });

            if (!response.ok) {
              throw new Error(`Upload failed: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('File uploaded successfully:', data);
            
            setUploadStatus('success');
            
            // Reset to idle after 3 seconds
            setTimeout(() => {
              setUploadStatus('idle');
              setUploadedFileName('');
            }, 3000);

          } catch (error) {
            console.error('Upload error:', error);
            setUploadStatus('error');
            setErrorMessage(error instanceof Error ? error.message : 'Upload failed');
            
            // Reset error after 5 seconds
            setTimeout(() => {
              setUploadStatus('idle');
              setErrorMessage('');
            }, 5000);
          }
        }
      }
    });
    
    el.click();
  };

  const getStatusIcon = () => {
    switch (uploadStatus) {
      case 'uploading':
        return <Loader2 className="animate-spin w-12 h-12" />;
      case 'success':
        return <CheckCircle className="w-12 h-12 text-green-400" />;
      case 'error':
        return <XCircle className="w-12 h-12 text-red-400" />;
      default:
        return <Upload className="w-12 h-12" />;
    }
  };

  const getStatusText = () => {
    switch (uploadStatus) {
      case 'uploading':
        return (
          <div className="text-center">
            <h3 className="text-lg font-semibold">Uploading...</h3>
            <p className="text-sm text-gray-300 mt-1">{uploadedFileName}</p>
          </div>
        );
      case 'success':
        return (
          <div className="text-center">
            <h3 className="text-lg font-semibold text-green-400">Upload Successful!</h3>
            <p className="text-sm text-gray-300 mt-1">{uploadedFileName}</p>
          </div>
        );
      case 'error':
        return (
          <div className="text-center">
            <h3 className="text-lg font-semibold text-red-400">Upload Failed</h3>
            <p className="text-sm text-gray-300 mt-1">{errorMessage}</p>
          </div>
        );
      default:
        return (
          <div className="text-center">
            <h3 className="text-lg font-semibold">Upload PDF File</h3>
            <p className="text-xs text-gray-400 mt-1">Max size: 10MB</p>
          </div>
        );
    }
  };

  return (
    <div className="w-full h-full flex items-center justify-center p-4">
      <div 
        className={`bg-slate-900 text-white shadow-2xl flex justify-center items-center p-8 rounded-lg border-2 transition-all duration-300 ${
          uploadStatus === 'idle' 
            ? 'border-white hover:border-blue-400 hover:shadow-blue-500/50 cursor-pointer' 
            : uploadStatus === 'success'
            ? 'border-green-400 shadow-green-500/50'
            : uploadStatus === 'error'
            ? 'border-red-400 shadow-red-500/50'
            : 'border-blue-400 shadow-blue-500/50'
        }`}
        onClick={uploadStatus === 'idle' ? handleFileUploadButtonClick : undefined}
      >
        <div className="flex justify-center items-center flex-col gap-4">
          {getStatusIcon()}
          {getStatusText()}
        </div>
      </div>
    </div>
  );
};

export default FileUploadComponent;