package main;

import java.util.Scanner;

import model.Chef;
import state.ChefIdle;

public class Main {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
				
		int input = -1;
		
		Chef chef = new Chef("Hooahwi", 12);
		
		do {
			System.out.println("1 . Order menu");
			System.out.println("2. exit");
			System.out.println(">> ");
			input =  sc.nextInt();sc.nextLine();
			
			switch (input) {
			case 1:
				
				if(chef.getState() instanceof ChefIdle) {
					
					do {
						chef.getState().voidchangeState();
					} while (chef.getState() instanceof ChefIdle == false);
				}
		
				break;

			default:
				break;
			}
			
		}while (input != 2);
		
}

}
