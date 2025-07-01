package spring.springai.domain.openai.service;

import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
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
}