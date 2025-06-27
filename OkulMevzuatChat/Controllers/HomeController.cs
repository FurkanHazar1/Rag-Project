using Microsoft.AspNetCore.Mvc;
using OkulMevzuatChat.Models;
using System.Text.Json;
using System.Text;

namespace OkulMevzuatChat.Controllers
{
    public class HomeController : Controller
    {
        private readonly HttpClient _httpClient;
        private readonly IConfiguration _configuration;
        private readonly ILogger<HomeController> _logger;

        public HomeController(HttpClient httpClient, IConfiguration configuration, ILogger<HomeController> logger)
        {
            _httpClient = httpClient;
            _configuration = configuration;
            _logger = logger;
            
            // Timeout'u kaldır - yavaş model için
            _httpClient.Timeout = TimeSpan.FromMilliseconds(-1); // Infinite timeout
        }

        public async Task<IActionResult> Index()
        {
            var viewModel = new ChatViewModel();
            
            // Python API durumunu kontrol et
            var apiBaseUrl = _configuration["PythonApi:BaseUrl"];
            try
            {
                var healthResponse = await _httpClient.GetAsync($"{apiBaseUrl}/health");
                if (!healthResponse.IsSuccessStatusCode)
                {
                    viewModel.ErrorMessage = "AI sistemi şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.";
                }
            }
            catch (Exception)
            {
                viewModel.ErrorMessage = "AI sistemi bağlantısı kurulamadı. Lütfen sistem yöneticisine başvurun.";
            }

            // Hoş geldin mesajı
            viewModel.Messages.Add(new ChatMessage
            {
                Content = "Merhaba! Size okul mevzuatı ile ilgili sorularınızda yardımcı olabilirim. Sorunuzu yazabilirsiniz.",
                IsUser = false,
                Timestamp = DateTime.Now
            });

            return View(viewModel);
        }

        [HttpPost]
        public async Task<IActionResult> SendMessage([FromBody] ChatRequest request)
        {
            if (!ModelState.IsValid)
            {
                return Json(new { success = false, error = "Geçersiz soru formatı" });
            }

            var apiBaseUrl = _configuration["PythonApi:BaseUrl"];
            
            try
            {
                // Python API'sine POST isteği gönder
                var options = new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                };
                var json = JsonSerializer.Serialize(request, options);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                _logger.LogInformation("Python API'sine soru gönderiliyor: {Question}", request.Question);
                _logger.LogInformation("⏳ Model yavaş olabilir, lütfen bekleyin...");

                var response = await _httpClient.PostAsync($"{apiBaseUrl}/chat", content);

                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync();
                    var chatResponse = JsonSerializer.Deserialize<ChatResponse>(responseContent, new JsonSerializerOptions 
                    { 
                        PropertyNameCaseInsensitive = true 
                    });

                    _logger.LogInformation("API'den cevap alındı, süre: {ResponseTime}s", chatResponse?.ResponseTime);

                    if (chatResponse != null && chatResponse.Success)
                    {
                        return Json(new
                        {
                            success = true,
                            answer = chatResponse.Answer,
                            responseTime = chatResponse.ResponseTime,
                            conversationId = chatResponse.ConversationId
                        });
                    }
                    else
                    {
                        return Json(new
                        {
                            success = false,
                            error = chatResponse?.ErrorMessage ?? "Bilinmeyen API hatası"
                        });
                    }
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError("Python API hatası: {StatusCode} - {Error}", response.StatusCode, errorContent);
                    
                    return Json(new
                    {
                        success = false,
                        error = $"API isteği başarısız oldu: {response.StatusCode}"
                    });
                }
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError("HTTP bağlantı hatası: {Error}", ex.Message);
                return Json(new
                {
                    success = false,
                    error = "Python API'sine bağlanılamadı. API'nin çalıştığından emin olun."
                });
            }
            catch (Exception ex)
            {
                _logger.LogError("Beklenmeyen hata: {Error}", ex.Message);
                return Json(new
                {
                    success = false,
                    error = "Beklenmeyen bir hata oluştu."
                });
            }
        }

        [HttpGet]
        public async Task<IActionResult> CheckHealth()
        {
            var apiBaseUrl = _configuration["PythonApi:BaseUrl"];
            try
            {
                var response = await _httpClient.GetAsync($"{apiBaseUrl}/health");
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var health = JsonSerializer.Deserialize<HealthResponse>(content, new JsonSerializerOptions 
                    { 
                        PropertyNameCaseInsensitive = true 
                    });
                    return Json(health);
                }
                return Json(new { status = "unhealthy", models_loaded = false });
            }
            catch
            {
                return Json(new { status = "connection_error", models_loaded = false });
            }
        }
    }
}