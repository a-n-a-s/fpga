//============================================================================
// Early Exit Controller - Decide whether to skip remaining layers
// Enables adaptive inference: exit early for high-confidence predictions
// For FPGA AI Accelerator with Efficient Inference features
//============================================================================

module early_exit_controller #(
    parameter CONF_THRESHOLD = 8'd180   // Confidence threshold for early exit (180/255 = 70%)
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 confidence_valid,  // Confidence data is valid
    input  wire [7:0]           confidence,        // Confidence score (0-255)
    input  wire                 high_confidence,   // High confidence flag
    output reg                  early_exit_en,     // Early exit enable (1 = skip remaining layers)
    output reg                  exit_taken,        // Early exit was taken
    output reg                  done               // Decision complete
);

    // State encoding
    reg [1:0] state;
    localparam S_IDLE    = 2'd0;
    localparam S_CHECK   = 2'd1;
    localparam S_DECIDE  = 2'd2;

    //========================================================================
    // Sequential Early Exit Decision Logic
    //========================================================================

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            early_exit_en <= 1'b0;
            exit_taken    <= 1'b0;
            done          <= 1'b0;
            state         <= S_IDLE;
        end
        else begin
            done <= 1'b0;  // Default: clear done

            case (state)
                S_IDLE: begin
                    if (confidence_valid) begin
                        state <= S_CHECK;
                    end
                end

                S_CHECK: begin
                    // Check if confidence is high enough for early exit
                    if (high_confidence && (confidence >= CONF_THRESHOLD)) begin
                        early_exit_en <= 1'b1;
                        exit_taken    <= 1'b1;
                    end
                    else begin
                        early_exit_en <= 1'b0;
                        exit_taken    <= 1'b0;
                    end
                    state <= S_DECIDE;
                end

                S_DECIDE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
