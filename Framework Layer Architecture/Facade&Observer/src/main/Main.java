package main;

import java.util.ArrayList;
import java.util.Scanner;

import facade.FakerFacade;
import model.Order;

public class Main {
	Scanner sc = new Scanner(System.in);
	FakerFacade command = new FakerFacade();
	public static void main(String[] args) {
		new Main();
	}
	public Main() {
		int choose;
		do {
			System.out.println("Menu");
			System.out.println("1. Place Order: ");
			System.out.println("2. View All Order: ");
			System.out.println("3. Exit");
			System.out.print("Input your choice: ");
			choose = sc.nextInt();sc.nextLine();
			
			switch (choose) {
			case 1:
				placeOrder();
				break;
			case 2:
				viewOrder();
				break;
			case 3:
				System.out.println("Thankyou");
				return;
			default:
				break;
			}
		}while(choose != 3);
	}
	
	private void placeOrder() {
		String name, orderItem;
		do {
			System.out.println("Input Your Name: ");
			name = sc.nextLine();
		} while (name.length() < 3);
		do {
			System.out.println("Input the item name: ");
			orderItem = sc.nextLine();
		} while (!(orderItem.equalsIgnoreCase("Burger") || orderItem.equalsIgnoreCase("Pizza") || orderItem.equalsIgnoreCase("Nasi Goreng")));
		
		//facade pattern
		command.placeOrder(name, orderItem);
	}
	
	private void viewOrder() {
		// TODO Auto-generated method stub
		ArrayList<Order> orders = command.getAllOrders();
		if(orders.isEmpty()) {
			System.out.println("No Order Placed");
		}
		else {
			for(Order order : orders) {
				System.out.println("Name: "+order.getUsername());
				System.out.println("Item ordered: " + order.getOrderItem());
			}
		}
			
	}
}
