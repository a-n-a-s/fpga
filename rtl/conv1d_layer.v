//============================================================================
// Conv1D Layer - 1D Convolution with ReLU activation
// Sequential filter processing for DSP efficiency
// Supports weight loading from ROM
//============================================================================

module conv1d_layer #(
    parameter DATA_WIDTH    = 8,              // Input/Output data width
    parameter ACC_WIDTH     = 32,             // Accumulator width
    parameter INPUT_LENGTH  = 128,            // Input sequence length
    parameter NUM_FILTERS   = 8,              // Number of filters
    parameter KERNEL_SIZE   = 3,              // Convolution kernel size
    parameter WEIGHT_ROM_DEPTH = 24           // conv1: 8 filters * 3 kernel = 24
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,          // Start convolution
    output reg                  done,           // Convolution complete
    
    // Streaming input
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                 valid_in,
    
    // Streaming output
    output reg signed [DATA_WIDTH-1:0] data_out,
    output reg                  valid_out,
    
    // Weight interface (from ROM)
    output reg [9:0]          weight_addr,    // Weight ROM address
    input  wire signed [DATA_WIDTH-1:0] weight_data,  // Weight from ROM
    output reg signed [ACC_WIDTH-1:0] bias_data     // Bias for current filter (32-bit)
);

    // Output length: INPUT_LENGTH - KERNEL_SIZE + 1
    localparam OUTPUT_LENGTH = INPUT_LENGTH - KERNEL_SIZE + 1;  // 126
    localparam WEIGHTS_PER_FILTER = KERNEL_SIZE;                // 3

    // Internal ROM for weights (synthesis will infer ROM)
    reg signed [DATA_WIDTH-1:0] weight_rom [0:WEIGHT_ROM_DEPTH-1];
    reg signed [ACC_WIDTH-1:0] bias_rom [0:NUM_FILTERS-1];  // 32-bit for INT32 biases

    // Sliding window outputs
    wire signed [DATA_WIDTH-1:0] win_s0, win_s1, win_s2;
    wire window_valid;
    wire sliding_done;
    
    // MAC unit signals
    reg mac_start;
    wire mac_done;
    wire signed [ACC_WIDTH-1:0] mac_acc;
    
    // State machine
    reg [2:0] state;
    localparam S_IDLE    = 3'd0;
    localparam S_INIT    = 3'd1;
    localparam S_CONV    = 3'd2;
    localparam S_RELU    = 3'd3;
    localparam S_NEXT_F  = 3'd4;
    localparam S_DONE    = 3'd5;
    
    // Counters and registers
    reg [2:0] filter_cnt;       // Current filter (0-7)
    reg [7:0] weight_idx;       // Weight index within filter (0-2)
    reg [7:0] output_idx;       // Output position counter
    reg [7:0] output_cnt;       // Outputs generated for current filter
    reg [7:0] total_out_cnt;    // Total outputs generated
    reg running;
    reg first_filter;
    reg signed [ACC_WIDTH-1:0] sum_val;  // Temporary sum for ReLU/saturation
    
    // Sliding window instance
    sliding_window_1d #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_LENGTH(INPUT_LENGTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .NUM_FILTERS(NUM_FILTERS)
    ) u_sliding_window (
        .clk(clk),
        .rst(rst),
        .start(running && (state == S_CONV)),
        .done(sliding_done),
        .data_in(data_in),
        .valid_in(valid_in),
        .win_sample0(win_s0),
        .win_sample1(win_s1),
        .win_sample2(win_s2),
        .window_valid(window_valid)
    );
    
    // MAC unit instance
    mac_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .NUM_STEPS(KERNEL_SIZE)
    ) u_mac (
        .clk(clk),
        .rst(rst),
        .start(mac_start),
        .done(mac_done),
        .data_in(win_s0),  // Use sample0 as data input
        .weight_in(weight_data),
        .data_valid(window_valid),
        .acc_out(mac_acc)
    );
    
    // Weight ROM initialization
    initial begin
        $readmemh("conv1_weights.mem", weight_rom);
        $readmemh("conv1_bias.mem", bias_rom);
    end

    //========================================================================
    // Sequential Conv1D with ReLU
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            running     <= 1'b0;
            filter_cnt  <= 3'd0;
            weight_idx  <= 8'd0;
            output_idx  <= 8'd0;
            output_cnt  <= 8'd0;
            total_out_cnt <= 8'd0;
            mac_start   <= 1'b0;
            weight_addr <= 10'd0;
            bias_data   <= {ACC_WIDTH{1'b0}};
            data_out    <= {DATA_WIDTH{1'b0}};
            valid_out   <= 1'b0;
            first_filter <= 1'b1;
        end
        else begin
            done      <= 1'b0;
            mac_start <= 1'b0;
            valid_out <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    if (start) begin
                        running     <= 1'b1;
                        state       <= S_INIT;
                        first_filter <= 1'b1;
                    end
                end
                
                S_INIT: begin
                    // Initialize for first filter
                    filter_cnt    <= 3'd0;
                    output_cnt    <= 8'd0;
                    weight_idx    <= 8'd0;
                    weight_addr   <= 10'd0;
                    bias_data     <= bias_rom[0];
                    state         <= S_CONV;
                end
                
                S_CONV: begin
                    if (window_valid && weight_idx < KERNEL_SIZE) begin
                        // Start MAC for current weight
                        mac_start   <= 1'b1;
                        weight_addr <= weight_addr + 10'd1;
                        weight_idx  <= weight_idx + 8'd1;
                    end
                    
                    if (mac_done) begin
                        // MAC complete, check if filter done
                        if (weight_idx == KERNEL_SIZE - 1) begin
                            // Last weight for this output position
                            state <= S_RELU;
                        end
                    end
                end
                
                S_RELU: begin
                    // Apply ReLU: max(0, acc + bias) - bias is already 32-bit
                    begin
                        sum_val = mac_acc + bias_data;
                        
                        // ReLU: clamp negative to zero
                        if (sum_val < 0) begin
                            data_out <= {DATA_WIDTH{1'b0}};
                        end
                        else begin
                            // Saturate to INT8 range
                            if (sum_val > 127)
                                data_out <= 8'd127;
                            else if (sum_val < -128)
                                data_out <= 8'd128;
                            else
                                data_out <= sum_val[DATA_WIDTH-1:0];
                        end
                    end
                    valid_out <= 1'b1;
                    output_cnt <= output_cnt + 8'd1;
                    
                    // Check if all outputs for this filter generated
                    if (output_cnt == OUTPUT_LENGTH - 1) begin
                        state <= S_NEXT_F;
                    end
                    else begin
                        // Reset for next output position
                        weight_idx <= 8'd0;
                        weight_addr <= filter_cnt * KERNEL_SIZE;
                        state <= S_CONV;
                    end
                end
                
                S_NEXT_F: begin
                    if (filter_cnt < NUM_FILTERS - 1) begin
                        // Move to next filter
                        filter_cnt  <= filter_cnt + 3'd1;
                        output_cnt  <= 8'd0;
                        weight_idx  <= 8'd0;
                        weight_addr <= (filter_cnt + 3'd1) * KERNEL_SIZE;
                        bias_data   <= bias_rom[filter_cnt + 3'd1];
                        state       <= S_CONV;
                    end
                    else begin
                        // All filters complete
                        state <= S_DONE;
                    end
                end
                
                S_DONE: begin
                    running <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
