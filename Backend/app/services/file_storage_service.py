"""
File Storage Management Service
Handles temporary file storage, cleanup, and organization of generated 3D models.
"""

import os
import shutil
import logging
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


class FileStorageService:
    """Service for managing file storage and cleanup operations"""
    
    def __init__(self, 
                 base_upload_dir: str = "./uploads",
                 base_generated_dir: str = "./generated",
                 temp_dir: str = "./temp"):
        """
        Initialize file storage service.
        
        Args:
            base_upload_dir: Directory for uploaded files
            base_generated_dir: Directory for generated models
            temp_dir: Directory for temporary files
        """
        self.base_upload_dir = Path(base_upload_dir)
        self.base_generated_dir = Path(base_generated_dir)
        self.temp_dir = Path(temp_dir)
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        # Configuration
        self.max_temp_age_hours = 24
        self.max_upload_age_days = 7
        self.max_generated_age_days = 30
    
    def _ensure_directories(self):
        """Create required directories"""
        directories = [
            self.base_upload_dir,
            self.base_generated_dir,
            self.temp_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def save_uploaded_file(self, file_content: bytes, original_filename: str) -> Dict[str, Any]:
        """
        Save an uploaded file to the upload directory.
        
        Args:
            file_content: File content as bytes
            original_filename: Original filename
            
        Returns:
            Dictionary with file information
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(file_content).hexdigest()[:8]
            stem = Path(original_filename).stem
            extension = Path(original_filename).suffix
            unique_filename = f"{timestamp}_{file_hash}_{stem}{extension}"
            
            # Save file
            file_path = self.base_upload_dir / unique_filename
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Generate file info
            file_stat = file_path.stat()
            file_info = {
                'filename': unique_filename,
                'original_filename': original_filename,
                'path': str(file_path),
                'size': file_stat.st_size,
                'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'hash': file_hash,
                'content_type': self._detect_content_type(extension)
            }
            
            logger.info(f"Saved uploaded file: {unique_filename} ({file_stat.st_size} bytes)")
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            raise
    
    def save_generated_model(self, model_data: bytes, model_name: str, format_ext: str = '.glb') -> Dict[str, Any]:
        """
        Save a generated 3D model file.
        
        Args:
            model_data: Model data as bytes
            model_name: Name for the model
            format_ext: File extension/format
            
        Returns:
            Dictionary with model file information
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(model_data).hexdigest()[:8]
            safe_name = "".join(c for c in model_name if c.isalnum() or c in " _-").strip()
            unique_filename = f"{timestamp}_{file_hash}_{safe_name}{format_ext}"
            
            # Save file
            file_path = self.base_generated_dir / unique_filename
            with open(file_path, 'wb') as f:
                f.write(model_data)
            
            # Generate model info
            file_stat = file_path.stat()
            model_info = {
                'filename': unique_filename,
                'model_name': model_name,
                'path': str(file_path),
                'size': file_stat.st_size,
                'format': format_ext,
                'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'hash': file_hash
            }
            
            logger.info(f"Saved generated model: {unique_filename} ({file_stat.st_size} bytes)")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to save generated model: {str(e)}")
            raise
    
    def create_temp_file(self, content: bytes, suffix: str = '') -> Dict[str, Any]:
        """
        Create a temporary file.
        
        Args:
            content: File content as bytes
            suffix: File suffix/extension
            
        Returns:
            Dictionary with temp file information
        """
        try:
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                dir=self.temp_dir
            )
            
            # Write content
            temp_file.write(content)
            temp_file.close()
            
            # Generate file info
            file_path = Path(temp_file.name)
            file_stat = file_path.stat()
            
            temp_info = {
                'path': str(file_path),
                'size': file_stat.st_size,
                'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=self.max_temp_age_hours)).isoformat()
            }
            
            logger.debug(f"Created temporary file: {file_path}")
            return temp_info
            
        except Exception as e:
            logger.error(f"Failed to create temporary file: {str(e)}")
            raise
    
    def get_uploaded_files(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get list of uploaded files.
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        try:
            files_info = []
            
            # Get all files in upload directory
            for file_path in self.base_upload_dir.iterdir():
                if file_path.is_file():
                    file_stat = file_path.stat()
                    file_info = {
                        'filename': file_path.name,
                        'path': str(file_path),
                        'size': file_stat.st_size,
                        'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        'modified_at': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    }
                    files_info.append(file_info)
            
            # Sort by creation time (newest first) and limit
            files_info.sort(key=lambda x: x['created_at'], reverse=True)
            return files_info[:limit]
            
        except Exception as e:
            logger.error(f"Failed to list uploaded files: {str(e)}")
            return []
    
    def get_generated_models(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get list of generated models.
        
        Args:
            limit: Maximum number of models to return
            
        Returns:
            List of model information dictionaries
        """
        try:
            models_info = []
            
            # Get all files in generated directory
            for file_path in self.base_generated_dir.iterdir():
                if file_path.is_file():
                    file_stat = file_path.stat()
                    model_info = {
                        'filename': file_path.name,
                        'path': str(file_path),
                        'size': file_stat.st_size,
                        'format': file_path.suffix.lower(),
                        'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        'modified_at': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    }
                    models_info.append(model_info)
            
            # Sort by creation time (newest first) and limit
            models_info.sort(key=lambda x: x['created_at'], reverse=True)
            return models_info[:limit]
            
        except Exception as e:
            logger.error(f"Failed to list generated models: {str(e)}")
            return []
    
    def cleanup_old_files(self) -> Dict[str, int]:
        """
        Clean up old temporary and uploaded files.
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'temp_files_deleted': 0,
            'upload_files_deleted': 0,
            'generated_files_deleted': 0,
            'total_space_freed': 0
        }
        
        try:
            # Cleanup temporary files
            temp_cutoff = datetime.now() - timedelta(hours=self.max_temp_age_hours)
            stats['temp_files_deleted'], temp_space = self._cleanup_directory(
                self.temp_dir, temp_cutoff
            )
            stats['total_space_freed'] += temp_space
            
            # Cleanup old uploaded files
            upload_cutoff = datetime.now() - timedelta(days=self.max_upload_age_days)
            upload_deleted, upload_space = self._cleanup_directory(
                self.base_upload_dir, upload_cutoff
            )
            stats['upload_files_deleted'] = upload_deleted
            stats['total_space_freed'] += upload_space
            
            # Cleanup old generated files
            generated_cutoff = datetime.now() - timedelta(days=self.max_generated_age_days)
            generated_deleted, generated_space = self._cleanup_directory(
                self.base_generated_dir, generated_cutoff
            )
            stats['generated_files_deleted'] = generated_deleted
            stats['total_space_freed'] += generated_space
            
            logger.info(f"Cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return stats
    
    def _cleanup_directory(self, directory: Path, cutoff_time: datetime) -> tuple[int, int]:
        """
        Clean up files in a directory older than cutoff time.
        
        Args:
            directory: Directory to clean
            cutoff_time: Cutoff datetime for file age
            
        Returns:
            Tuple of (files_deleted, space_freed_in_bytes)
        """
        deleted_count = 0
        space_freed = 0
        
        try:
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_modified < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        space_freed += file_size
                        logger.debug(f"Deleted old file: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {str(e)}")
        
        return deleted_count, space_freed
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage usage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            def get_directory_stats(directory: Path) -> Dict[str, Any]:
                total_size = 0
                file_count = 0
                oldest_file = None
                newest_file = None
                
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        file_stat = file_path.stat()
                        total_size += file_stat.st_size
                        file_count += 1
                        
                        file_time = datetime.fromtimestamp(file_stat.st_mtime)
                        if oldest_file is None or file_time < oldest_file:
                            oldest_file = file_time
                        if newest_file is None or file_time > newest_file:
                            newest_file = file_time
                
                return {
                    'total_size': total_size,
                    'file_count': file_count,
                    'oldest_file': oldest_file.isoformat() if oldest_file else None,
                    'newest_file': newest_file.isoformat() if newest_file else None
                }
            
            stats = {
                'upload_directory': get_directory_stats(self.base_upload_dir),
                'generated_directory': get_directory_stats(self.base_generated_dir),
                'temp_directory': get_directory_stats(self.temp_dir),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate totals
            stats['totals'] = {
                'total_size': sum(dir_stats['total_size'] for dir_stats in [
                    stats['upload_directory'], stats['generated_directory'], stats['temp_directory']
                ]),
                'total_files': sum(dir_stats['file_count'] for dir_stats in [
                    stats['upload_directory'], stats['generated_directory'], stats['temp_directory']
                ])
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {}
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a specific file.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                file_size = path.stat().st_size
                path.unlink()
                logger.info(f"Deleted file: {file_path} ({file_size} bytes)")
                return True
            else:
                logger.warning(f"File not found or not a file: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False
    
    def _detect_content_type(self, extension: str) -> str:
        """Detect content type based on file extension"""
        content_types = {
            '.pdf': 'application/pdf',
            '.eml': 'message/rfc822',
            '.txt': 'text/plain',
            '.obj': 'model/obj',
            '.stl': 'model/stl',
            '.ply': 'model/ply',
            '.glb': 'model/gltf-binary',
            '.gltf': 'model/gltf+json'
        }
        return content_types.get(extension.lower(), 'application/octet-stream')


# Global service instance
storage_service = FileStorageService()


def get_storage_service() -> FileStorageService:
    """Get the global storage service instance"""
    return storage_service


# Example usage and testing
if __name__ == "__main__":
    service = FileStorageService()
    
    # Test file saving
    test_content = b"This is a test file content"
    print("Saving test file...")
    file_info = service.save_uploaded_file(test_content, "test.txt")
    print(f"Saved: {file_info}")
    
    # Test storage stats
    print("\nStorage statistics:")
    stats = service.get_storage_stats()
    print(f"Total files: {stats['totals']['total_files']}")
    print(f"Total size: {stats['totals']['total_size']} bytes")
    
    # Test cleanup
    print("\nRunning cleanup...")
    cleanup_stats = service.cleanup_old_files()
    print(f"Cleanup results: {cleanup_stats}")