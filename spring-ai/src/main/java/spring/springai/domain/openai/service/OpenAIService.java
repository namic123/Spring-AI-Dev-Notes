package spring.springai.domain.openai.service;

import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.openai.*;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

import java.util.List;

@Service
@FieldDefaults(makeFinal = true)
@RequiredArgsConstructor
public class OpenAIService {
    // OpenAI에서 제공하는 다양한 모델 컴포넌트들
    private OpenAiChatModel openAiChatModel;
    private OpenAiEmbeddingModel openAiEmbeddingModel;
    private OpenAiImageModel openAiImageModel;
    private OpenAiAudioSpeechModel openAiAudioSpeechModel;
    private OpenAiAudioTranscriptionModel openAiAudioTranscriptionModel;

    /**
     * OpenAI Chat 모델을 사용하여 단일 응답을 생성하는 메서드
     * @param text 사용자 입력 텍스트
     * @return OpenAI 응답 텍스트 (단일 응답)
     */
    public String generate(String text){

        // 역할별 메시지 객체 생성
        SystemMessage systemMessage = new SystemMessage(""); // 시스템 역할: 일반적으로 지시문 입력
        UserMessage userMessage = new UserMessage(text);     // 사용자 입력 메시지
        AssistantMessage assistantMessage = new AssistantMessage(""); // (빈) 이전 AI 응답, 컨텍스트 유지용

        // OpenAI 옵션 설정: 모델 이름, 온도 등
        OpenAiChatOptions options = OpenAiChatOptions.builder()
                .model("gpt-4.1-mini") // 사용할 모델명
                .temperature(0.7)      // 생성 다양성 조절
                .build();

        // 메시지와 옵션으로 Prompt 구성
        Prompt prompt = new Prompt(List.of(systemMessage, userMessage, assistantMessage), options);

        // Chat 모델 호출 및 결과 반환
        ChatResponse response = openAiChatModel.call(prompt);
        return response.getResult().getOutput().getText(); // 최종 응답 텍스트 추출
    }

    /**
     * OpenAI Chat 모델을 사용하여 스트리밍 방식으로 응답을 받는 메서드
     * @param text 사용자 입력 텍스트
     * @return Flux<String> 형태로 응답을 실시간 스트리밍
     */
    public Flux<String> generateStream(String text) {

        // 역할별 메시지 구성 (generate()와 동일)
        SystemMessage systemMessage = new SystemMessage("");
        UserMessage userMessage = new UserMessage(text);
        AssistantMessage assistantMessage = new AssistantMessage("");

        // Chat 옵션 설정
        OpenAiChatOptions options = OpenAiChatOptions.builder()
                .model("gpt-4.1-mini")
                .temperature(0.7)
                .build();

        // Prompt 구성
        Prompt prompt = new Prompt(List.of(systemMessage, userMessage, assistantMessage), options);

        // 스트리밍 API 호출: 응답이 Flux<ChatResponse> 형태로 수신됨
        return openAiChatModel.stream(prompt)
                .mapNotNull(response -> response.getResult().getOutput().getText()); // 텍스트 추출 및 반환
    }

    /**
     * 주어진 텍스트 리스트에 대해 OpenAI Embedding 모델을 사용하여 벡터 임베딩을 생성
     * <p>
     * 각 텍스트는 OpenAI 모델에 의해 float 배열 형태의 벡터로 변환되며,
     * 이 벡터들은 자연어 처리 및 RAG 기반 검색 등에 활용될 수 있음.
     *
     * @param texts 임베딩을 생성할 입력 텍스트 목록
     * @param model 사용할 OpenAI 임베딩 모델 이름 (예: "text-embedding-3-small")
     * @return 각 텍스트에 대응하는 임베딩 벡터 리스트 (float 배열)
     */
    public List<float[]> generateEmbedding(List<String> texts, String model) {

        // 1. 사용할 임베딩 모델 이름 설정
        EmbeddingOptions embeddingOptions = OpenAiEmbeddingOptions.builder()
                .model(model) // ex: "text-embedding-3-small"
                .build();

        // 2. 입력 텍스트와 옵션을 EmbeddingRequest 객체로 래핑
        EmbeddingRequest prompt = new EmbeddingRequest(texts, embeddingOptions);

        // 3. OpenAI Embedding 모델에 요청을 보내고 응답을 받음
        EmbeddingResponse response = openAiEmbeddingModel.call(prompt);

        // 4. 응답으로부터 임베딩 벡터(float[])만 추출하여 리스트로 반환
        return response.getResults().stream()
                .map(Embedding::getOutput) // 각 Embedding 객체에서 float[] 벡터 추출
                .toList();
    }
}