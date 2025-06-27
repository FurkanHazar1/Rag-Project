// Models/ChatModels.cs
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace OkulMevzuatChat.Models
{
    public class ChatRequest
    {
        [Required(ErrorMessage = "Soru alanı boş olamaz")]
        [StringLength(1000, ErrorMessage = "Soru en fazla 1000 karakter olabilir")]
        [JsonPropertyName("question")]
        public string Question { get; set; } = string.Empty;
        
        [JsonPropertyName("conversation_id")]
        public string? ConversationId { get; set; }
    }

    public class ChatResponse
    {
        [JsonPropertyName("answer")]
        public string Answer { get; set; } = string.Empty;
        
        [JsonPropertyName("conversation_id")]
        public string ConversationId { get; set; } = string.Empty;
        
        [JsonPropertyName("response_time")]
        public double ResponseTime { get; set; }
        
        [JsonPropertyName("success")]
        public bool Success { get; set; }
        
        [JsonPropertyName("error_message")]
        public string? ErrorMessage { get; set; }
    }

    public class ChatMessage
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string Content { get; set; } = string.Empty;
        public bool IsUser { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.Now;
        public bool IsLoading { get; set; } = false;
    }

    public class ChatViewModel
    {
        public List<ChatMessage> Messages { get; set; } = new List<ChatMessage>();
        public string CurrentQuestion { get; set; } = string.Empty;
        public bool IsLoading { get; set; } = false;
        public string? ErrorMessage { get; set; }
    }

    public class HealthResponse
    {
        public string Status { get; set; } = string.Empty;
        public bool ModelsLoaded { get; set; }
        public string Version { get; set; } = string.Empty;
        public string? ErrorMessage { get; set; }
    }
}