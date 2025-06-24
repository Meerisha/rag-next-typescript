import { Agent, run } from "@openai/agents";
import { openai } from "@ai-sdk/openai";
import { aisdk } from "@openai/agents-extensions";
import { VectorizeService } from "@/lib/vectorize";
import type { ChatSource } from "@/types/chat";

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    const userMessage = messages[messages.length - 1];
    let contextDocuments = "";
    let sources: ChatSource[] = [];

    // Retrieve relevant documents using Vectorize if there's a user message
    if (userMessage?.role === "user" && userMessage?.content) {
      try {
        const vectorizeService = new VectorizeService();
        const documents = await vectorizeService.retrieveDocuments(
          userMessage.content
        );
        contextDocuments =
          vectorizeService.formatDocumentsForContext(documents);
        sources = vectorizeService.convertDocumentsToChatSources(documents);
      } catch (vectorizeError) {
        console.error("Vectorize retrieval failed:", vectorizeError);
        contextDocuments =
          "Unable to retrieve relevant documents at this time.";
        sources = [];
      }
    }

    // Create a model instance using the AI SDK adapter
    const model = aisdk(openai("gpt-4o-mini"));

    // Create an agent with enhanced instructions that include RAG context
    const agent = new Agent({
      name: "RAG Assistant Agent",
      instructions: `You are a helpful AI assistant that specializes in answering questions based on provided context documents.

${contextDocuments ? `
=== CONTEXT DOCUMENTS ===
${contextDocuments}
=== END CONTEXT DOCUMENTS ===

Please base your responses primarily on the context provided above when relevant. If the context doesn't contain information to answer the question, acknowledge this and provide general knowledge while being clear about what information comes from the context vs. your general knowledge.
` : "No specific context documents are available for this query. Please provide helpful general information."}

Keep your answers concise and informative, limited to 10 sentences or fewer.
Be conversational and engaging while maintaining accuracy.`,
      model,
    });

    // Format the conversation history for the agent
    const conversationMessages = messages.map((msg: any) => {
      if (msg.role === "user") {
        return msg.content;
      } else if (msg.role === "assistant") {
        return `Assistant: ${msg.content}`;
      }
      return "";
    }).filter(Boolean).join("\n\n");

    // Get the latest user question
    const latestUserQuestion = userMessage?.content || "Hello";

    // Run the agent with the conversation context
    const result = await run(
      agent,
      `Previous conversation context:\n${conversationMessages.length > latestUserQuestion.length ? conversationMessages : ""}\n\nCurrent question: ${latestUserQuestion}`
    );

    // Extract the text from the agent's response
    const responseText = typeof result === 'string' ? result : 
                        result?.messages?.[result.messages.length - 1]?.content || 
                        "I apologize, but I couldn't generate a proper response.";

    // Return both the text response and sources in the same format as the original endpoint
    return Response.json({
      role: "assistant",
      content: responseText,
      sources: sources,
      agent: "OpenAI Agents SDK", // Add identifier to show this came from the agents endpoint
    });
  } catch (error) {
    console.error("Error in agents chat:", error);
    return Response.json(
      { 
        error: "Failed to process agents chat",
        details: error instanceof Error ? error.message : "Unknown error"
      }, 
      { status: 500 }
    );
  }
} 